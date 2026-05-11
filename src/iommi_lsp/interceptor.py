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
from .analyzers.base import Analyzer, CompletionResult, Diagnostic


_log = log.get("interceptor")

PUBLISH_DIAGNOSTICS = "textDocument/publishDiagnostics"
INITIALIZE = "initialize"
DID_OPEN = "textDocument/didOpen"
DID_CHANGE = "textDocument/didChange"
DID_SAVE = "textDocument/didSave"
DID_CLOSE = "textDocument/didClose"
COMPLETION = "textDocument/completion"


class DocumentStore:
    """In-memory mirror of the editor's open buffers.

    LSP's text-sync notifications (``didOpen``/``didChange``/``didClose``)
    carry the *unsaved* document content; the file on disk lags the
    editor by however long it's been since the user hit save. Analyzers
    that want to validate what the user is *currently* looking at have
    to read from this store, not from disk.
    """

    def __init__(self) -> None:
        self._docs: dict[str, str] = {}

    def get(self, uri: str) -> str | None:
        return self._docs.get(uri)

    def did_open(self, uri: str, text: str) -> None:
        self._docs[uri] = text

    def did_change(self, uri: str, content_changes: list[dict]) -> None:
        text = self._docs.get(uri)
        for change in content_changes:
            if "range" not in change:
                # Full-document sync — replaces everything.
                text = change.get("text", "")
                continue
            if text is None:
                # Incremental change against a buffer we never saw open.
                # Shouldn't happen with a well-behaved client; bail.
                _log.warning("incremental change for unknown uri %s; dropping", uri)
                return
            text = _apply_incremental_change(text, change)
        if text is not None:
            self._docs[uri] = text

    def did_close(self, uri: str) -> None:
        self._docs.pop(uri, None)


def _apply_incremental_change(text: str, change: dict) -> str:
    rng = change["range"]
    start = _offset_from_lsp_position(text, rng["start"])
    end = _offset_from_lsp_position(text, rng["end"])
    return text[:start] + change.get("text", "") + text[end:]


def _offset_from_lsp_position(text: str, pos: dict) -> int:
    """Convert LSP ``{line, character}`` to a Python ``str`` offset.

    LSP characters are UTF-16 code units by default (positionEncoding).
    Non-BMP code points (e.g. emoji) count as 2 UTF-16 units. For ASCII
    Python source — the overwhelming common case — this collapses to
    straight character indexing.
    """
    target_line = int(pos.get("line", 0))
    target_char = int(pos.get("character", 0))

    offset = 0
    line = 0
    n = len(text)
    while offset < n and line < target_line:
        if text[offset] == "\n":
            line += 1
        offset += 1

    char_units = 0
    while offset < n and char_units < target_char:
        ch = text[offset]
        if ch == "\n":
            break
        char_units += 2 if ord(ch) > 0xFFFF else 1
        offset += 1
    return offset


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
        added = self._added(uri)

        _log.debug(
            "publishDiagnostics uri=%s in=%d kept=%d dropped=%d added=%d",
            uri,
            len(diagnostics),
            len(kept),
            len(diagnostics) - len(kept),
            len(added),
        )

        if len(kept) == len(diagnostics) and not added:
            return body

        params["diagnostics"] = kept + added
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

    def _added(self, uri: str) -> list[Diagnostic]:
        out: list[Diagnostic] = []
        for a in self.analyzers:
            adder = getattr(a, "additional_diagnostics", None)
            if adder is None:
                continue
            try:
                out.extend(adder(uri))
            except Exception:
                _log.exception("analyzer %s additional_diagnostics crashed", getattr(a, "name", a))
        return out


def _ensure_completion_capability(payload: dict, original_body: bytes) -> bytes:
    """Patch an ``initialize`` response so the editor knows we offer
    completions. If ty already advertises ``completionProvider`` we
    leave the payload alone; otherwise we add a minimal entry. The
    matchmaker's ``on_response`` handler is what actually fills the
    items at request time.
    """
    result = payload.get("result")
    if not isinstance(result, dict):
        return original_body
    caps = result.get("capabilities")
    if not isinstance(caps, dict):
        caps = {}
        result["capabilities"] = caps
    if "completionProvider" in caps:
        return original_body
    caps["completionProvider"] = {
        "triggerCharacters": ["(", ","],
        "resolveProvider": False,
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


class CompletionMatchmaker:
    """Two-sided hook that augments completion responses with analyzer items.

    Editor → ty: captures every ``textDocument/completion`` request's
    id alongside its uri/position. The body is forwarded unchanged.

    ty → editor: when a response carries an id we've captured, we
    compute the analyzers' completions for that uri/position and merge
    them into ``result.items`` (handling both the list and
    ``CompletionList`` response shapes). Unmatched responses pass
    through untouched, so this is zero-cost when no analyzer is
    interested.
    """

    def __init__(self, analyzers: Sequence[Analyzer] = ()) -> None:
        self.analyzers: list[Analyzer] = list(analyzers)
        self._pending: dict[Any, tuple[str, dict]] = {}
        self._pending_initialize: set[Any] = set()

    async def on_request(self, body: bytes) -> bytes | None:
        if not body or body[:1] != b"{":
            return body
        try:
            payload: Any = json.loads(body)
        except json.JSONDecodeError:
            return body
        if not isinstance(payload, dict):
            return body
        method = payload.get("method")
        msg_id = payload.get("id")
        if method == INITIALIZE and msg_id is not None:
            self._pending_initialize.add(msg_id)
            return body
        if method != COMPLETION:
            return body
        if msg_id is None:
            # Notifications shouldn't carry textDocument/completion, but
            # tolerate the malformed case by forwarding silently.
            return body
        params = payload.get("params") or {}
        doc = (params.get("textDocument") or {})
        uri = doc.get("uri")
        position = params.get("position")
        if isinstance(uri, str) and isinstance(position, dict):
            self._pending[msg_id] = (uri, position)
        return body

    async def on_response(self, body: bytes) -> bytes | None:
        if not body or body[:1] != b"{":
            return body
        try:
            payload: Any = json.loads(body)
        except json.JSONDecodeError:
            return body
        if not isinstance(payload, dict):
            return body
        msg_id = payload.get("id")
        if msg_id is None:
            return body
        if msg_id in self._pending_initialize:
            self._pending_initialize.discard(msg_id)
            return _ensure_completion_capability(payload, body)
        context = self._pending.pop(msg_id, None)
        if context is None:
            return body
        uri, position = context
        extras, exclusive = self._gather(uri, position)
        if not extras and not exclusive:
            return body

        # If ty errored on completion (e.g. it doesn't implement the
        # method), swap the error for a success containing our items —
        # otherwise the editor would surface ty's error and discard the
        # whole response.
        if "error" in payload and "result" not in payload:
            payload.pop("error", None)
            payload["result"] = {
                "isIncomplete": False,
                "items": list(extras),
            }
            return json.dumps(payload, separators=(",", ":")).encode("utf-8")

        result = payload.get("result")
        if exclusive:
            # Drop whatever ty produced — at an ORM-kwarg position its
            # free-form variable completions are noise. ``isIncomplete``
            # is False so the editor stops asking for more.
            payload["result"] = {"isIncomplete": False, "items": list(extras)}
        elif result is None:
            payload["result"] = {"isIncomplete": False, "items": list(extras)}
        elif isinstance(result, list):
            payload["result"] = result + list(extras)
        elif isinstance(result, dict):
            existing = list(result.get("items") or [])
            existing.extend(extras)
            result["items"] = existing
            payload["result"] = result
        else:
            return body   # unexpected shape; don't touch

        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def _gather(self, uri: str, position: dict) -> tuple[list[dict], bool]:
        """Collect items + exclusivity across analyzers.

        Accepts both the structured ``CompletionResult`` return and the
        legacy bare ``list[dict]`` — the latter is treated as
        non-exclusive so older analyzer code keeps working.
        """
        all_items: list[dict] = []
        exclusive = False
        for a in self.analyzers:
            fn = getattr(a, "completions", None)
            if fn is None:
                continue
            try:
                result = fn(uri, position)
            except Exception:
                _log.exception(
                    "analyzer %s completions crashed",
                    getattr(a, "name", a),
                )
                continue
            if isinstance(result, CompletionResult):
                all_items.extend(result.items)
                exclusive = exclusive or result.exclusive
            elif isinstance(result, list):
                all_items.extend(result)
        return all_items, exclusive


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
        document_store: DocumentStore | None = None,
    ) -> None:
        self._on_workspace = on_workspace
        self._on_file_changed = on_file_changed
        self._document_store = document_store
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
        elif method == DID_OPEN:
            self._handle_did_open(payload)
        elif method == DID_CHANGE:
            self._handle_did_change(payload)
        elif method == DID_CLOSE:
            self._handle_did_close(payload)
        elif method == DID_SAVE and self._on_file_changed:
            doc = (payload.get("params") or {}).get("textDocument") or {}
            uri = _file_uri_or_none(doc.get("uri"))
            if uri:
                self._spawn(self._on_file_changed(uri))
        return body

    def _handle_did_open(self, payload: dict) -> None:
        params = payload.get("params") or {}
        doc = params.get("textDocument") or {}
        uri = _file_uri_or_none(doc.get("uri"))
        if not uri:
            return
        if self._document_store is not None:
            self._document_store.did_open(uri, doc.get("text") or "")

    def _handle_did_change(self, payload: dict) -> None:
        params = payload.get("params") or {}
        doc = params.get("textDocument") or {}
        uri = _file_uri_or_none(doc.get("uri"))
        if not uri:
            return
        if self._document_store is not None:
            changes = params.get("contentChanges") or []
            self._document_store.did_change(uri, list(changes))
        if self._on_file_changed is not None:
            self._spawn(self._on_file_changed(uri))

    def _handle_did_close(self, payload: dict) -> None:
        params = payload.get("params") or {}
        doc = params.get("textDocument") or {}
        uri = _file_uri_or_none(doc.get("uri"))
        if not uri:
            return
        if self._document_store is not None:
            self._document_store.did_close(uri)

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.create_task(_swallow(coro))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


async def _swallow(coro: Awaitable[None]) -> None:
    try:
        await coro
    except Exception:
        _log.exception("background sniffer callback failed")
