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

from . import jsonrpc, log
from .analyzers.base import Analyzer, CompletionResult, Diagnostic


_log = log.get("interceptor")

PUBLISH_DIAGNOSTICS = "textDocument/publishDiagnostics"
INITIALIZE = "initialize"
DID_OPEN = "textDocument/didOpen"
DID_CHANGE = "textDocument/didChange"
DID_SAVE = "textDocument/didSave"
DID_CLOSE = "textDocument/didClose"
COMPLETION = "textDocument/completion"
CANCEL_REQUEST = "$/cancelRequest"


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


def _items_of_result(result: Any) -> list | None:
    """Return the mutable item list inside a completion *result*, or None.

    LSP allows either a bare ``list`` of ``CompletionItem`` or a
    ``CompletionList`` (``{"isIncomplete": …, "items": [...]}``). Returns
    the underlying list (mutating it mutates the payload), or None for
    anything we don't recognise.
    """
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        items = result.get("items")
        if isinstance(items, list):
            return items
    return None


def _annotate_sort_text(items: list, partial: str) -> None:
    """Rank items whose match-text starts with *partial* ahead of the rest.

    Sets ``sortText`` on each item AND reorders the list in place so the
    client surfaces prefix matches first — e.g. typing ``fi`` at
    ``User.objects.fi`` puts ``filter`` / ``first`` above ``afirst`` /
    ``complex_filter``. Some LSP clients (TUI plugins, older protocols)
    ignore ``sortText`` and just display server-side order; others do
    their own fuzzy scoring and use ``sortText`` only as a tiebreaker.
    Doing both — reorder the array AND set sortText — wins in all
    cases. Within each priority bucket we preserve whatever order the
    producer (ty or an analyzer) intended by suffixing the item's
    existing ``sortText`` (or, failing that, its label). Case-insensitive:
    editors fold case for prefix matching by default and our own kwarg
    labels are all lowercase anyway.
    """
    if not partial:
        return
    partial_lc = partial.lower()
    for item in items:
        if not isinstance(item, dict):
            continue
        match_text = item.get("filterText") or item.get("label") or ""
        if not isinstance(match_text, str):
            continue
        existing = item.get("sortText")
        base = existing if isinstance(existing, str) else match_text
        priority = "0" if match_text.lower().startswith(partial_lc) else "1"
        item["sortText"] = f"{priority}_{base}"

    def _key(it):
        if not isinstance(it, dict):
            return "2_"
        st = it.get("sortText")
        return st if isinstance(st, str) else "2_"
    items.sort(key=_key)


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
        "triggerCharacters": ["(", ",", "/"],
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

    When a ``text_provider`` is supplied, we also annotate every item in
    the merged response with a ``sortText`` that ranks prefix-of-cursor
    matches above non-prefix matches — so typing ``fi`` at
    ``User.objects.fi`` surfaces ``filter`` / ``first`` ahead of
    ``afirst`` / ``complex_filter``.
    """

    def __init__(
        self,
        analyzers: Sequence[Analyzer] = (),
        text_provider: Callable[[str], str | None] | None = None,
    ) -> None:
        self.analyzers: list[Analyzer] = list(analyzers)
        self._text_provider = text_provider
        self._pending: dict[Any, tuple[str, dict]] = {}
        self._pending_initialize: set[Any] = set()
        # Ids the editor told us to cancel before ty's response arrived.
        # We bound the set so a misbehaving client can't grow it unboundedly
        # (cancel-without-response would leak otherwise).
        self._cancelled: set[Any] = set()
        self._cancelled_cap = 1024
        # When configured, short-circuit known-exclusive completion
        # positions (settings literals, etc.) by writing the response
        # directly to the editor instead of round-tripping through ty.
        # ty's completion latency in large settings.py files dominates
        # perceived sluggishness — and ty's contribution there is noise
        # (free-form variable names inside a string literal), so skipping
        # it is both faster *and* more correct.
        self._editor_writer: asyncio.StreamWriter | None = None

    def attach_editor_writer(self, writer: asyncio.StreamWriter) -> None:
        """Wire up the direct-to-editor response path used for short-
        circuiting exclusive completions. Optional — when not set, the
        matchmaker behaves exactly like before (round-trip through ty
        and augment in :meth:`on_response`)."""
        self._editor_writer = writer

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
        if method == CANCEL_REQUEST:
            cancel_id = (payload.get("params") or {}).get("id")
            if cancel_id is not None and cancel_id in self._pending:
                self._pending.pop(cancel_id, None)
                if len(self._cancelled) >= self._cancelled_cap:
                    # Drop the oldest entry — set order is insertion order
                    # in CPython, so this is the one that's been waiting
                    # longest without a response from ty.
                    self._cancelled.discard(next(iter(self._cancelled)))
                self._cancelled.add(cancel_id)
            return body
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
        if not (isinstance(uri, str) and isinstance(position, dict)):
            return body

        # Short-circuit known-exclusive positions: when an analyzer
        # claims a position outright (Django ORM kwarg, INSTALLED_APPS,
        # iommi auto field name…), ty's response is going to be
        # discarded anyway — so don't wait for it. Build the response
        # here and write it directly to the editor.
        if self._editor_writer is not None:
            try:
                extras, exclusive, incomplete = self._gather(uri, position)
            except Exception:
                _log.exception("short-circuit gather crashed; falling back to ty")
                extras, exclusive, incomplete = [], False, True
            if exclusive:
                partial = self._partial_at(uri, position)
                items = list(extras)
                _annotate_sort_text(items, partial)
                result = {
                    "isIncomplete": not (exclusive and not incomplete),
                    "items": items,
                }
                synth = json.dumps(
                    {"jsonrpc": "2.0", "id": msg_id, "result": result},
                    separators=(",", ":"),
                ).encode("utf-8")
                try:
                    await jsonrpc.write_message(self._editor_writer, synth)
                except Exception:
                    _log.exception("synthetic completion write failed; falling back")
                else:
                    return None   # don't forward to ty; we've already responded
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
        if msg_id in self._cancelled:
            # Editor already cancelled this; forward ty's reply (likely a
            # RequestCancelled error) untouched and skip analyzers. Never
            # rewrite an error into a synthetic success — the editor has
            # moved on.
            self._cancelled.discard(msg_id)
            self._pending.pop(msg_id, None)
            return body
        if msg_id in self._pending_initialize:
            self._pending_initialize.discard(msg_id)
            return _ensure_completion_capability(payload, body)
        context = self._pending.pop(msg_id, None)
        if context is None:
            return body
        uri, position = context
        extras, exclusive, incomplete = self._gather(uri, position)
        partial = self._partial_at(uri, position)

        # Decide what kind of mutation (if any) we need to make.
        merged = bool(extras) or exclusive
        # When we know the buffer text, we always want to repack so we can
        # set ``isIncomplete`` deliberately — that prevents the editor
        # from serving a stale cached completion list with its own
        # scoring while the user keeps typing (which is what keeps
        # ``afirst`` at the top once you've reached ``User.objects.fi``).
        # The exception is when every analyzer that contributed says its
        # items are complete (e.g. INSTALLED_APPS) — there we want
        # ``isIncomplete: false`` so the editor caches and filters
        # locally instead of round-tripping every keystroke.
        will_repack = self._text_provider is not None
        if not merged and not will_repack:
            return body   # zero-copy: nothing for us to do here

        original_result = payload.get("result")

        # If ty errored on completion (e.g. it doesn't implement the
        # method), swap the error for a success containing our items —
        # otherwise the editor would surface ty's error and discard the
        # whole response.
        if "error" in payload and "result" not in payload:
            if not extras:
                # Nothing to substitute with — leave the error alone.
                return body
            payload.pop("error", None)
            payload["result"] = {"isIncomplete": True, "items": list(extras)}
        elif exclusive:
            # Drop whatever ty produced — at an ORM-kwarg position its
            # free-form variable completions are noise.
            payload["result"] = {"isIncomplete": True, "items": list(extras)}
        elif extras:
            if original_result is None:
                payload["result"] = {"isIncomplete": True, "items": list(extras)}
            elif isinstance(original_result, list):
                payload["result"] = original_result + list(extras)
            elif isinstance(original_result, dict):
                existing = list(original_result.get("items") or [])
                existing.extend(extras)
                original_result["items"] = existing
                payload["result"] = original_result
            else:
                return body   # unexpected shape; don't touch

        if will_repack:
            items = _items_of_result(payload.get("result"))
            if items is not None:
                _annotate_sort_text(items, partial)
            result = payload.get("result")
            if isinstance(result, dict):
                # Only force re-request when re-querying is actually
                # required — either because we're augmenting ty's items
                # (so our prefix-priority sort needs the latest partial)
                # or because at least one contributing analyzer says its
                # items depend on context that the editor's local filter
                # can't reproduce (e.g. iommi ``__`` chain crossings).
                result["isIncomplete"] = not (exclusive and not incomplete)

        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def _partial_at(self, uri: str, position: dict) -> str:
        if self._text_provider is None:
            return ""
        text = self._text_provider(uri)
        if not isinstance(text, str):
            return ""
        offset = _offset_from_lsp_position(text, position)
        end = min(offset, len(text))
        start = end
        while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
            start -= 1
        return text[start:end]

    def _gather(self, uri: str, position: dict) -> tuple[list[dict], bool, bool]:
        """Collect items + exclusivity + incompleteness across analyzers.

        Accepts both the structured ``CompletionResult`` return and the
        legacy bare ``list[dict]`` — the latter is treated as
        non-exclusive so older analyzer code keeps working.

        Stops the moment an analyzer returns ``exclusive=True``: by
        contract that analyzer owns the position and no other analyzer
        is going to contribute anything but empty/non-exclusive (each
        recognises a different context — INSTALLED_APPS string vs. ORM
        kwarg vs. iommi auto field). Running the rest just to confirm
        they have nothing to say costs an AST parse each. For a
        16 KB ``settings.py`` that's ~1.5 ms of pure waste per keystroke
        — enough to be felt on burst typing.

        The returned ``incomplete`` flag is the OR across analyzers that
        actually contributed (items or an exclusive empty). An analyzer
        that didn't speak up doesn't sway the decision either way; if
        nobody spoke up, ``incomplete`` defaults to True so we keep the
        re-query-on-every-keystroke behaviour for ty-only responses.
        """
        all_items: list[dict] = []
        exclusive = False
        incomplete = False
        spoke = False
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
                if result.items or result.exclusive:
                    spoke = True
                    incomplete = incomplete or result.incomplete
                if result.exclusive:
                    exclusive = True
                    break   # this analyzer claims the position; skip the rest
            elif isinstance(result, list):
                all_items.extend(result)
                if result:
                    spoke = True
                    incomplete = True
        if not spoke:
            incomplete = True
        return all_items, exclusive, incomplete


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
        # Deliberately *don't* fire on_file_changed here. didChange fires
        # per keystroke and the disk file hasn't moved — the Django
        # analyzer's reindex (the expensive bit, ~120 ms on a workspace
        # with hundreds of models) would re-scrape the same on-disk
        # content every keystroke and rebuild the whole index for
        # nothing. The DocumentStore is enough to keep completion fresh;
        # disk-state reindexing only needs to happen on didSave (which
        # the main dispatcher above handles).

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
