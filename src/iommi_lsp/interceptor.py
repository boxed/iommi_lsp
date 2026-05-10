"""Editor↔ty hooks: workspace-init sniffing and diagnostic filtering.

The proxy installs this as the ``ty→editor`` hook. For every frame:

* If it's not parseable JSON or not a ``textDocument/publishDiagnostics``
  notification, forward unchanged (the common case — no allocation cost
  beyond a single ``json.loads``).
* Otherwise, run each diagnostic through the registered analyzers'
  ``is_false_positive`` predicate. Survivors are kept; if **any** analyzer
  flags a diagnostic, it's dropped.
* If the surviving list equals the original, forward the original bytes
  unchanged — this avoids an unnecessary re-serialization on the hot
  path when no filtering happens (which will be ~all messages until we
  flip on the Django filter).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from . import log
from .analyzers.base import Analyzer, Diagnostic


_log = log.get("interceptor")

PUBLISH_DIAGNOSTICS = "textDocument/publishDiagnostics"
INITIALIZE = "initialize"
DID_CHANGE = "textDocument/didChange"
DID_SAVE = "textDocument/didSave"


class DiagnosticInterceptor:
    """Stateful hook for the ``ty→editor`` direction."""

    def __init__(self, analyzers: Sequence[Analyzer] = ()) -> None:
        self.analyzers: list[Analyzer] = list(analyzers)

    async def __call__(self, body: bytes) -> bytes | None:
        # Cheap reject path: only JSON-object frames could be diagnostics.
        # ``ty`` always sends well-formed JSON-RPC, but we stay defensive.
        if not body or body[:1] != b"{":
            return body

        try:
            payload: Any = json.loads(body)
        except json.JSONDecodeError:
            _log.warning("could not parse ty→editor frame as JSON; forwarding raw")
            return body

        if not isinstance(payload, dict) or payload.get("method") != PUBLISH_DIAGNOSTICS:
            return body

        params = payload.get("params") or {}
        uri = params.get("uri", "")
        diagnostics: list[Diagnostic] = list(params.get("diagnostics") or [])

        kept = self._filter(uri, diagnostics)

        _log.debug(
            "publishDiagnostics uri=%s in=%d kept=%d dropped=%d",
            uri,
            len(diagnostics),
            len(kept),
            len(diagnostics) - len(kept),
        )

        if len(kept) == len(diagnostics):
            # No analyzer wanted to drop anything → forward verbatim.
            return body

        params["diagnostics"] = kept
        payload["params"] = params
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def _filter(self, uri: str, diagnostics: list[Diagnostic]) -> list[Diagnostic]:
        if not self.analyzers:
            return diagnostics
        kept: list[Diagnostic] = []
        for diag in diagnostics:
            if any(a.is_false_positive(uri, diag) for a in self.analyzers):
                continue
            kept.append(diag)
        return kept


# ---------------------------------------------------------------------------
# Editor → ty hook: workspace sniffing and file-change notifications.
# ---------------------------------------------------------------------------


WorkspaceCallback = Callable[[Path], Awaitable[None]]
ChangeCallback = Callable[[str], Awaitable[None]]


def _workspace_root_from_initialize(payload: dict) -> Path | None:
    params = payload.get("params") or {}
    folders = params.get("workspaceFolders")
    if isinstance(folders, list) and folders:
        first = folders[0]
        uri = (first or {}).get("uri") if isinstance(first, dict) else None
        path = _file_uri_to_path(uri)
        if path is not None:
            return path
    root_uri = params.get("rootUri")
    path = _file_uri_to_path(root_uri)
    if path is not None:
        return path
    root_path = params.get("rootPath")
    if isinstance(root_path, str) and root_path:
        return Path(root_path)
    return None


def _file_uri_to_path(uri: Any) -> Path | None:
    if not isinstance(uri, str) or not uri.startswith("file://"):
        return None
    parsed = urlparse(uri)
    return Path(unquote(parsed.path))


def _file_uri_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.startswith("file://") else None


class EditorRequestSniffer:
    """Watches editor → ty traffic for workspace-init and file-change events.

    Forwards every frame untouched; the side effects (kicking off
    workspace indexing, invalidating per-file caches) happen in the
    background so they never block the message pump.
    """

    def __init__(
        self,
        *,
        on_workspace: WorkspaceCallback | None = None,
        on_file_changed: ChangeCallback | None = None,
    ) -> None:
        self._on_workspace = on_workspace
        self._on_file_changed = on_file_changed
        self._workspace_seen = False
        self._tasks: set[asyncio.Task[None]] = set()

    async def __call__(self, body: bytes) -> bytes | None:
        if body[:1] != b"{":
            return body
        try:
            payload: Any = json.loads(body)
        except json.JSONDecodeError:
            return body
        if not isinstance(payload, dict):
            return body
        method = payload.get("method")
        if method == INITIALIZE and not self._workspace_seen and self._on_workspace:
            root = _workspace_root_from_initialize(payload)
            if root is not None:
                self._workspace_seen = True
                self._spawn(self._on_workspace(root))
        elif method in (DID_CHANGE, DID_SAVE) and self._on_file_changed:
            doc = (payload.get("params") or {}).get("textDocument") or {}
            uri = _file_uri_or_none(doc.get("uri"))
            if uri:
                self._spawn(self._on_file_changed(uri))
        return body

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.create_task(_swallow(coro))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


async def _swallow(coro: Awaitable[None]) -> None:
    try:
        await coro
    except Exception:
        _log.exception("background sniffer callback failed")
