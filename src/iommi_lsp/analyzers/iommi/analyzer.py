"""IommiAnalyzer — adds diagnostics for invalid ``Class(kw__chain=...)``.

Loads the workspace's ``.iommi_lsp-graph.json`` (produced by
``iommi_lsp graph build``) and validates each call whose callee is a
known iommi class. The first dead-end segment in a flattened kwarg
chain becomes a ``unknown-iommi-refinable`` diagnostic at that
segment's source range.

The analyzer never *removes* diagnostics — it only adds — so it composes
cleanly with the Django filter on the same proxy.
"""

from __future__ import annotations

import ast
import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from typing import TYPE_CHECKING

from ... import log
from ..base import CompletionResult, Diagnostic
from ..django import lookup_walker
from .graph import GRAPH_FILENAME, IommiClass, IommiGraph, Refinable, load_graph
from .walker import (
    Problem,
    _ATTRS_NAME,
    _all_refinables,
    _resolve_refinable,
    _synthetic_html_attrs,
    walk,
)

if TYPE_CHECKING:
    from ..django.index import DjangoIndex, ModelInfo


_log = log.get("iommi.analyzer")


_IOMMI_DIAG_CODE = "iommi-unknown-refinable"
_IOMMI_DIAG_SOURCE = "iommi_lsp"
_IOMMI_ATTR_PATH_CODE = "iommi-unknown-attr-path"
_IOMMI_CALLABLE_CODE = "iommi-callable-expected"


# Refinable names at chain leaves that iommi calls as functions. Passing
# a string here is almost always a typo for a name reference (the classic
# ``Action(post_handler='save')`` instead of ``post_handler=save``).
# Kept narrow on purpose — overbroad flagging masks more bugs than it
# catches, since some refinables (``extra__url`` and friends) are happy
# with strings.
_CALLABLE_LEAVES: frozenset[str] = frozenset({
    "post_handler",   # Action(post_handler=fn) / Form(actions__x__post_handler=fn)
    "func",           # endpoints__<name>__func=view, Page(parts__handler__func=fn)
    "on_commit",      # form lifecycle hooks
    "on_save",
})


class _Unset:
    pass


_UNSET = _Unset()


# Built-in iommi styles. Users can register more via
# ``register_style(...)``; we don't try to discover those at index time
# (would require importing user code). When a style isn't in this set
# we stay non-exclusive so custom styles aren't suppressed.
_IOMMI_BUILTIN_STYLES: tuple[str, ...] = (
    "base",
    "base_enhanced_forms",
    "bootstrap",
    "bootstrap5",
    "bulma",
    "daisyui",
    "django_admin",
    "font_awesome_4",
    "font_awesome_6",
    "foundation",
    "select2_enhanced_forms",
    "semantic_ui",
    "uikit",
    "us_web_design_system",
    "vanilla_css",
    "water",
)


@dataclass
class _ParsedFile:
    tree: ast.Module
    source: str


class IommiAnalyzer:
    name = "iommi"

    def __init__(
        self,
        workspace_root: Path,
        graph: IommiGraph | None = None,
        text_provider: Callable[[str], str | None] | None = None,
        django_index_provider: "Callable[[], DjangoIndex | None] | None" = None,
        auto_build: bool = True,
    ) -> None:
        self.workspace_root = workspace_root
        self.graph: IommiGraph = graph or IommiGraph()
        self._text_provider = text_provider
        # iommi's ``auto__model=Model``/``auto__rows=qs`` pattern lets us
        # surface a model's fields as ``columns__<field>`` completions.
        # The provider is a callable rather than a snapshot so the iommi
        # analyzer always sees the latest Django index without having to
        # be re-initialised when the index updates incrementally.
        self._django_index_provider = django_index_provider
        # Whether ``index()`` should attempt to build a missing graph
        # automatically. Tests that want to assert "no graph" behaviour
        # pass ``auto_build=False`` to disable the in-process / subprocess
        # build attempts.
        self._auto_build = auto_build
        self._cache: dict[str, _ParsedFile] = {}
        # Background rebuild handle. ``index()`` spawns this whenever a
        # graph was loaded from disk so the user benefits from the
        # latest reflector logic (and any iommi upgrade since the graph
        # was last written) without having to rebuild manually. Stored
        # on ``self`` so tests can await it.
        self._rebuild_task: asyncio.Task | None = None

    # -- Analyzer protocol ----------------------------------------------------

    async def index(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        graph_path = workspace_root / GRAPH_FILENAME
        loaded = load_graph(graph_path)
        if loaded is not None:
            self.graph = loaded
            _log.info(
                "loaded iommi graph: %d classes (iommi %s)",
                len(self.graph.classes), self.graph.iommi_version,
            )
            self._cache.clear()
            # Schedule a background rebuild. The on-disk graph can be
            # stale in two ways the schema_version bump doesn't catch:
            # the user upgraded iommi without rebuilding, or the
            # reflector itself learned new tricks (e.g. picking up
            # ``@refinable``-decorated methods) in a newer iommi_lsp.
            # Rebuilding in the background means we use the loaded
            # graph immediately and swap in the fresh one atomically
            # when it's ready.
            if self._auto_build:
                self._rebuild_task = asyncio.create_task(
                    self._background_rebuild(graph_path)
                )
            return

        # No graph on disk. Try to build one — first in-process (free
        # when iommi_lsp shares a venv with iommi), then fall back to
        # synthesized stubs for the well-known iommi classes. Each step
        # logs its outcome so the user can see exactly what went wrong.
        if self._auto_build:
            built = await asyncio.to_thread(_try_build_graph, graph_path)
        else:
            built = None
            _log.info("no iommi graph at %s (auto_build disabled)", graph_path)
        if built is not None:
            self.graph = built
        else:
            self.graph = IommiGraph()
            _log.info(
                "iommi completions will use synthesized stubs for known "
                "classes (Table/Form/Page/Query). Project-specific subclasses "
                "and exact refinable shapes won't be available until the graph "
                "is built."
            )
        self._cache.clear()

    async def _background_rebuild(self, graph_path: Path) -> None:
        """Rebuild the graph and atomically swap it in if successful.

        Runs the reflector in a thread so a slow subprocess build
        doesn't block the event loop. On failure (iommi not importable
        from any candidate interpreter, broken reflector, etc.) we keep
        the previously loaded graph — never worse than where we
        started. Single-attribute store to ``self.graph`` is atomic in
        CPython, so synchronous readers (``additional_diagnostics`` /
        ``completions``) see either the old or the new graph but never
        a torn one.
        """
        _log.info("scheduling background iommi graph refresh for %s", graph_path)
        try:
            built = await asyncio.to_thread(_try_build_graph, graph_path)
        except Exception:
            _log.exception("background iommi graph refresh crashed")
            return
        if built is None:
            return
        old = self.graph
        self.graph = built
        self._cache.clear()
        _log.info(
            "swapped in refreshed iommi graph: %d → %d classes "
            "(iommi %s → %s)",
            len(old.classes), len(built.classes),
            old.iommi_version, built.iommi_version,
        )

    async def on_file_changed(self, uri: str) -> None:
        self._cache.pop(uri, None)

    def is_false_positive(self, uri: str, diagnostic: Diagnostic) -> bool:
        return False  # we only add, never subtract

    def additional_diagnostics(self, uri: str) -> list[Diagnostic]:
        if not self.graph.classes:
            return []
        path = _uri_to_path(uri)
        if path is None:
            return []
        parsed = self._parse(uri, path)
        if parsed is None:
            return []
        return list(self._scan(parsed))

    def completions(self, uri: str, position: dict) -> CompletionResult:
        """Return refinable completions when the cursor sits inside an iommi
        class call's kwarg name. Mirrors the Django ORM-kwarg flow: we
        patch the buffer with a marker keyword so an in-progress call
        parses, then walk the graph to find what valid sub-segments are.

        Works even when no ``.iommi_lsp-graph.json`` has been built —
        the well-known iommi classes (Table/Form/Page/Query) carry a
        hardcoded stub so ``Table(auto__...)`` and ``Table(columns__...)``
        still get exclusive iommi completions and ty's variable noise
        stays suppressed.
        """
        empty = CompletionResult()
        path = _uri_to_path(uri)
        if path is None:
            return empty
        source = self._source_for(uri, path)
        if source is None:
            return empty
        try:
            return self._scan_completions(source, position)
        except Exception:
            _log.exception("iommi completion scanner crashed; emitting nothing")
            return empty

    def _scan_completions(self, source: str, position: dict) -> CompletionResult:
        empty = CompletionResult()
        line = int(position.get("line", 0))
        character = int(position.get("character", 0))
        offset = _offset_from_lsp_position(source, line, character)
        if offset > len(source):
            return empty

        # Inside an `auto__include=['…']` / `auto__exclude=['…']` string
        # literal, we want to suggest the auto-bound model's field names
        # as string values — that's a separate completion shape
        # (textual, not a kwarg name).
        string_ctx = _string_state_at(source, offset)
        if string_ctx is not None:
            return self._complete_in_string(source, offset, string_ctx)

        partial_start = offset
        while partial_start > 0 and (
            source[partial_start - 1].isalnum()
            or source[partial_start - 1] == "_"
        ):
            partial_start -= 1
        partial = source[partial_start:offset]

        # Cheap precondition — iommi kwarg names can only follow ``(``
        # or ``,``, OR sit at the start of a statement inside a
        # ``class Meta:`` body (iommi treats Meta assignments as kwargs
        # to the enclosing class's constructor). Skips ~13 ms of buffer
        # parse for top-level identifiers in big files.
        in_call = _is_call_arg_position(source, partial_start)
        in_meta = (not in_call) and _is_meta_assignment_position(
            source, partial_start,
        )
        if not (in_call or in_meta):
            return empty

        marker = "__iommi_lsp_completion_marker__"
        head = source[:partial_start]
        inserted = marker + "=None"
        closes = _close_brackets(head + inserted)
        patched = head + inserted + closes

        try:
            tree = ast.parse(patched)
        except SyntaxError:
            return empty

        imports = _collect_imports(tree)

        if in_call:
            marker_call = _find_marker_call(tree, marker)
            if marker_call is None:
                return empty
            cls_qualname = _resolve_callee(marker_call.func, imports)
            auto_model = self._resolve_auto_model(marker_call)
        else:
            meta_owner = _find_marker_meta_owner(tree, marker)
            if meta_owner is None:
                return empty
            user_iommi_subclasses = _collect_user_iommi_subclasses(
                tree, imports, self.graph,
            )
            cls_qualname = _resolve_iommi_base(
                meta_owner, imports, self.graph, user_iommi_subclasses,
            )
            if cls_qualname is None:
                # No iommi-known base, but a well-known iommi class
                # (Table/Form/Page/Query) is synthesisable downstream.
                # Mirror the call path, which hands ``_resolve_callee``'s
                # bare qualname to ``_synthesize_iommi_class`` so users
                # get exclusive completions before they've run
                # ``iommi_lsp graph build``.
                for base in meta_owner.bases:
                    qn = _resolve_callee(base, imports)
                    if qn is not None:
                        cls_qualname = qn
                        break
            meta_def = _find_meta_class(meta_owner)
            auto_model = (
                self._resolve_auto_model_from_meta(meta_def)
                if meta_def is not None else None
            )

        if cls_qualname is None:
            return empty
        cls = self.graph.get(cls_qualname)
        if cls is None:
            simple = cls_qualname.rsplit(".", 1)[-1]
            cls = self.graph.lookup_simple(simple)
        if cls is None:
            # No graph data (yet) — fall back to a hardcoded stub for
            # the well-known iommi classes. This keeps `Table(auto__...)`
            # and friends exclusive even before the user runs
            # ``iommi_lsp graph build``.
            cls = _synthesize_iommi_class(cls_qualname)
        if cls is None:
            return empty

        if "__" in partial:
            head_chain, _, suffix = partial.rpartition("__")
            chain = head_chain.split("__")
            items = _complete_after_chain(
                self.graph, cls, chain, suffix,
                prefix=head_chain + "__",
                auto_model=auto_model,
            )
        else:
            items = _complete_top_level(self.graph, cls, partial)

        if items is None:
            # Chain didn't resolve to something we can enumerate — but
            # we *are* inside an iommi kwarg slot, so still exclusive.
            return CompletionResult(items=[], exclusive=True)
        return CompletionResult(items=list(items), exclusive=True)

    def _complete_in_string(
        self, source: str, offset: int, ctx: "_StringCtx"
    ) -> CompletionResult:
        """Field-name completion inside ``auto__include`` / ``auto__exclude``.

        We close the open quote at the cursor, balance brackets, parse,
        then locate which (iommi-class) call's keyword owns the string
        at *offset*. Anything but the ``auto__include`` /
        ``auto__exclude`` keyword is non-exclusive empty so we don't
        suppress ty in unrelated string positions.
        """
        empty = CompletionResult()
        partial = source[ctx.start + 1:offset]
        head = source[:offset]
        # Close the open quote, then balance brackets/parens so ast
        # has a complete program to chew on.
        closes = ctx.quote + _close_brackets(head + ctx.quote)
        patched = head + closes
        try:
            tree = ast.parse(patched)
        except SyntaxError:
            return empty
        # Stash the patched source so `_node_offset_matches` can map an
        # ast (line, col) back to a source offset for comparison.
        setattr(tree, "_iommi_source", patched)

        # ``style='‸'`` — offer iommi's built-in style names.
        style_target = _find_style_string(tree, ctx.start)
        if style_target is not None:
            items: list[dict] = []
            for name in _IOMMI_BUILTIN_STYLES:
                if partial and not name.startswith(partial):
                    continue
                items.append({
                    "label": name,
                    "kind": 21,
                    "insertText": name,
                    "detail": "iommi style",
                    "data": {"source": "iommi_lsp.iommi-style"},
                })
            # Non-exclusive: users can register custom styles and we
            # don't want to hide those from ty's name completion.
            return CompletionResult(items=items, exclusive=False)

        # Find the iommi Call whose `auto__include`/`auto__exclude`
        # keyword contains a Constant string at the cursor position.
        target = _find_auto_include_string(tree, ctx.start)
        if target is None:
            return empty
        call, kw_arg, str_node = target

        imports = _collect_imports(tree)
        cls_qualname = _resolve_callee(call.func, imports)
        if cls_qualname is None:
            return empty
        cls = self.graph.get(cls_qualname)
        if cls is None:
            simple = cls_qualname.rsplit(".", 1)[-1]
            cls = self.graph.lookup_simple(simple)
        if cls is None:
            cls = _synthesize_iommi_class(cls_qualname)
        if cls is None:
            return empty

        auto_model = self._resolve_auto_model(call)
        if auto_model is None:
            return empty

        # The user is typing a bare field name inside the literal —
        # offer plain names (no `=`/`__` suffix).
        items: list[dict] = []
        for name in sorted(auto_model.fields):
            if partial and not name.startswith(partial):
                continue
            fi = auto_model.fields[name]
            items.append({
                "label": name,
                "kind": 5,
                "insertText": name,
                "detail": f"auto field ({fi.field_type}) on {auto_model.qualname}",
                "data": {
                    "source": "iommi_lsp.iommi-auto-field",
                    "class": cls.qualname,
                    "model": auto_model.qualname,
                    "kwarg": kw_arg,
                },
            })
        return CompletionResult(items=items, exclusive=True)

    def _resolve_auto_model(self, call: ast.Call) -> "ModelInfo | None":
        """Try to bind a Django ModelInfo via the call's ``auto__*`` kwargs.

        Recognises ``auto__model=Model`` (bare Name) and the queryset-
        ish forms ``auto__rows=Model.objects.<method>(...)`` /
        ``auto__instance=Model.objects.<method>(...)``. Anything more
        exotic is ignored — bias toward returning ``None`` so we don't
        invent field completions on the wrong model.
        """
        if self._django_index_provider is None:
            return None
        index = self._django_index_provider()
        if index is None or not getattr(index, "models", None):
            return None
        for kw in call.keywords:
            if kw.arg in ("auto__model", "model"):
                model = _resolve_model_from_name(kw.value, index)
                if model is not None:
                    return model
            elif kw.arg in ("auto__rows", "auto__instance", "rows", "instance"):
                model = _resolve_model_from_manager_chain(kw.value, index)
                if model is not None:
                    return model
        return None

    def _resolve_auto_model_from_meta(
        self, meta: ast.ClassDef,
    ) -> "ModelInfo | None":
        """Meta-body twin of ``_resolve_auto_model``. iommi treats
        ``class Meta: auto__model = User`` as the kwarg form ``auto__model=
        User`` on the enclosing class's constructor."""
        if self._django_index_provider is None:
            return None
        index = self._django_index_provider()
        if index is None or not getattr(index, "models", None):
            return None
        for stmt in meta.body:
            if isinstance(stmt, ast.AnnAssign):
                if not isinstance(stmt.target, ast.Name) or stmt.value is None:
                    continue
                pairs = [(stmt.target.id, stmt.value)]
            elif isinstance(stmt, ast.Assign):
                pairs = [
                    (t.id, stmt.value)
                    for t in stmt.targets if isinstance(t, ast.Name)
                ]
            else:
                continue
            for name, value in pairs:
                if name in ("auto__model", "model"):
                    model = _resolve_model_from_name(value, index)
                    if model is not None:
                        return model
                elif name in ("auto__rows", "auto__instance", "rows", "instance"):
                    model = _resolve_model_from_manager_chain(value, index)
                    if model is not None:
                        return model
        return None

    # -- internals ------------------------------------------------------------

    def _parse(self, uri: str, path: Path) -> _ParsedFile | None:
        source = self._source_for(uri, path)
        if source is None:
            return None
        cached = self._cache.get(uri)
        if cached is not None and cached.source == source:
            return cached
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            _log.debug("could not parse %s: %s", path, e)
            return None
        parsed = _ParsedFile(tree=tree, source=source)
        self._cache[uri] = parsed
        return parsed

    def _source_for(self, uri: str, path: Path) -> str | None:
        if self._text_provider is not None:
            text = self._text_provider(uri)
            if text is not None:
                return text
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            _log.debug("could not read %s: %s", path, e)
            return None

    def _scan(self, parsed: _ParsedFile):
        imports = _collect_imports(parsed.tree)
        user_iommi_subclasses = _collect_user_iommi_subclasses(
            parsed.tree, imports, self.graph,
        )
        for node in ast.walk(parsed.tree):
            if isinstance(node, ast.Call):
                yield from self._scan_call(parsed, node, imports)
            elif isinstance(node, ast.ClassDef):
                yield from self._scan_class_meta(
                    parsed, node, imports, user_iommi_subclasses,
                )

    def _scan_call(
        self, parsed: _ParsedFile, node: ast.Call, imports: dict[str, str],
    ):
        cls_qualname = _resolve_callee(node.func, imports)
        if cls_qualname is None:
            return
        cls = self.graph.get(cls_qualname)
        if cls is None:
            # Try simple-name lookup — useful when the user imports
            # a class via re-export (`from iommi import Table`) but
            # we recorded it under its source module.
            simple = cls_qualname.rsplit(".", 1)[-1]
            cls = self.graph.lookup_simple(simple)
            if cls is None:
                return

        auto_model: "ModelInfo | None | _Unset" = _UNSET
        for kw in node.keywords:
            if kw.arg is None:
                continue   # **kwargs splat — skip
            chain = kw.arg.split("__")
            result = walk(self.graph, cls.qualname, chain)
            if isinstance(result, Problem):
                diag = _problem_to_diagnostic(parsed.source, kw, chain, result)
                if diag is not None:
                    yield diag
                continue

            # ``Form(fields__name__attr='path')`` / ``Table(columns__c__attr='p')``
            # — bridge iommi's ``attr`` value to a Django model lookup. iommi
            # resolves the attr string against the bound auto model at
            # runtime; we can do the same statically when ``auto__model=``
            # (or ``rows=``/``instance=``/``model=``) is in the same call.
            if (
                len(chain) >= 3
                and chain[-1] == "attr"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                if auto_model is _UNSET:
                    auto_model = self._resolve_auto_model(node)
                if auto_model is not None:
                    diag = self._validate_attr_path(
                        kw.value, auto_model,
                    )
                    if diag is not None:
                        yield diag

            # ``Action(post_handler='save')`` / endpoint ``__func='view'`` —
            # iommi calls these values as functions; a string Constant is
            # almost always a typo for a name reference.
            if chain[-1] in _CALLABLE_LEAVES:
                diag = _validate_callable_value(kw)
                if diag is not None:
                    yield diag

    def _validate_attr_path(
        self, const: ast.Constant, model: "ModelInfo",
    ) -> Diagnostic | None:
        index = (
            self._django_index_provider()
            if self._django_index_provider else None
        )
        if index is None:
            return None
        value = const.value
        if not isinstance(value, str) or not value:
            return None
        chain = lookup_walker.split_chain(value)
        result = lookup_walker.walk(index, model.qualname, chain)
        if not isinstance(result, lookup_walker.Problem):
            return None
        if const.lineno is None or const.col_offset is None:
            return None
        line0 = const.lineno - 1
        col_start = const.col_offset + 1   # skip opening quote
        seg_offset = 0
        for i, seg in enumerate(chain):
            if i == result.segment_index:
                break
            seg_offset += len(seg) + len("__")
        col_start += seg_offset
        col_end = col_start + len(result.bad_segment)
        msg = (
            f"unknown attr path segment {result.bad_segment!r} "
            f"on {result.on_model}"
        )
        if result.available:
            hint = ", ".join(sorted(result.available)[:8])
            if len(result.available) > 8:
                hint += ", …"
            msg += f"  (available: {hint})"
        return {
            "code": _IOMMI_ATTR_PATH_CODE,
            "message": msg,
            "range": {
                "start": {"line": line0, "character": col_start},
                "end": {"line": line0, "character": col_end},
            },
            "severity": 2,
            "source": _IOMMI_DIAG_SOURCE,
        }

    def _scan_class_meta(
        self,
        parsed: _ParsedFile,
        node: ast.ClassDef,
        imports: dict[str, str],
        user_iommi_subclasses: dict[str, str],
    ):
        """Validate ``class Meta:`` attribute names against the enclosing
        iommi class's refinables. Per iommi's equivalency docs, names
        inside ``Meta`` are passed straight through to the constructor —
        ``class Meta: columns__name__after = 'x'`` is the same as
        ``Table(columns__name__after='x')``.
        """
        cls_qualname = _resolve_iommi_base(
            node, imports, self.graph, user_iommi_subclasses,
        )
        if cls_qualname is None:
            return
        meta = _find_meta_class(node)
        if meta is None:
            return
        for stmt in meta.body:
            for name_node in _meta_assignment_targets(stmt):
                chain = name_node.id.split("__")
                result = walk(self.graph, cls_qualname, chain)
                if isinstance(result, Problem):
                    yield _problem_at_name(name_node, chain, result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_build_graph(graph_path: Path) -> IommiGraph | None:
    """Attempt to build an iommi graph for the user's workspace.

    Strategy (each step logs detailed reasons on failure):

    1. **In-process build.** If ``iommi`` is importable in this Python
       process — i.e. iommi_lsp shares a venv with iommi — reflect
       directly. This is the cheap path and the one we recommend (install
       iommi_lsp with ``uv tool install --with iommi iommi_lsp`` or in
       the same venv as the project).
    2. **Project-venv subprocess.** Look for ``.venv/bin/python`` /
       ``venv/bin/python`` under the workspace and try ``iommi_lsp graph
       build`` against that interpreter. Requires ``iommi_lsp`` to be
       installed there too.

    Returns the built graph on success (and persists it to
    *graph_path*); returns ``None`` if every strategy failed. The
    caller falls back to synthesised stubs.
    """
    if graph_path.exists():
        _log.info("rebuilding iommi graph at %s", graph_path)
    else:
        _log.info("no iommi graph at %s — attempting to build", graph_path)

    # Strategy 1: in-process.
    inline = _try_inline_build(graph_path)
    if inline is not None:
        return inline

    # Strategy 2: project venv subprocess.
    workspace_root = graph_path.parent
    sub = _try_subprocess_build(workspace_root, graph_path)
    if sub is not None:
        return sub

    _log.warning(
        "could not build iommi graph automatically. To get full iommi "
        "completions:\n"
        "  - install iommi alongside iommi_lsp: `uv tool install --with iommi --force iommi_lsp`, OR\n"
        "  - run `iommi_lsp graph build --python <path-to-project-venv-python>` once "
        "from the project root, OR\n"
        "  - install `iommi_lsp` in the project venv and run `iommi_lsp graph build` from there."
    )
    return None


def _try_inline_build(graph_path: Path) -> IommiGraph | None:
    try:
        import iommi as _iommi   # noqa: F401
    except ImportError as e:
        _log.info(
            "in-process build skipped: iommi not importable here (%s). "
            "iommi_lsp's tool venv usually doesn't include iommi.", e,
        )
        return None
    iommi_version = getattr(_iommi, "__version__", "?")
    _log.info("in-process iommi import OK (version %s) — reflecting", iommi_version)
    try:
        from . import build as _build_mod  # avoid top-level cycle
        from .graph import save_graph as _save_graph
        from .reflect import build as _reflect_build

        graph = _reflect_build()
        _save_graph(graph, graph_path)
        _log.info(
            "built iommi graph in-process: %d classes → %s",
            len(graph.classes), graph_path,
        )
        return graph
    except Exception as e:   # broad on purpose: reflect can raise anything
        _log.warning("in-process iommi graph build failed: %s", e, exc_info=True)
        return None


def _try_subprocess_build(workspace_root: Path, graph_path: Path) -> IommiGraph | None:
    candidates = [
        workspace_root / ".venv" / "bin" / "python",
        workspace_root / "venv" / "bin" / "python",
        workspace_root / ".venv" / "Scripts" / "python.exe",
    ]
    pythons = [p for p in candidates if p.exists()]
    if not pythons:
        _log.info(
            "no workspace venv Python found (looked under %s). Skipping subprocess build.",
            workspace_root,
        )
        return None

    from .build import GraphBuildError, build_in_subprocess
    from .graph import save_graph as _save_graph

    for py in pythons:
        _log.info("attempting subprocess graph build with %s", py)
        try:
            graph = build_in_subprocess(python=str(py))
        except GraphBuildError as e:
            _log.warning("subprocess build with %s failed: %s", py, e)
            continue
        _save_graph(graph, graph_path)
        _log.info(
            "built iommi graph via %s: %d classes (iommi %s) → %s",
            py, len(graph.classes), graph.iommi_version, graph_path,
        )
        return graph
    return None


def _uri_to_path(uri: str) -> Path | None:
    if not uri.startswith("file://"):
        return None
    parsed = urlparse(uri)
    return Path(unquote(parsed.path))


def _collect_imports(tree: ast.Module) -> dict[str, str]:
    """Map local name → fully-qualified import. Same idea as the Django index."""
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")[0]
                out[alias.asname or head] = alias.name if alias.asname else head
        elif isinstance(node, ast.ImportFrom):
            if node.module is None or node.level:
                continue
            for alias in node.names:
                local = alias.asname or alias.name
                out[local] = f"{node.module}.{alias.name}"
    return out


def _resolve_callee(func: ast.AST, imports: dict[str, str]) -> str | None:
    """Resolve ``Class``, ``mod.Class``, ``a.b.Class`` to a qualname via imports."""
    if isinstance(func, ast.Name):
        return imports.get(func.id, func.id)
    if isinstance(func, ast.Attribute):
        flat = _flatten_attribute(func)
        if flat is None:
            return None
        head, _, tail = flat.partition(".")
        if head in imports:
            return f"{imports[head]}.{tail}" if tail else imports[head]
        return flat
    return None


def _flatten_attribute(node: ast.AST) -> str | None:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _collect_user_iommi_subclasses(
    tree: ast.Module, imports: dict[str, str], graph: IommiGraph,
) -> dict[str, str]:
    """Build a name → iommi-base-qualname map for user-declared classes.

    Walks class defs in source order so a chain like
    ``class A(Table); class B(A): class Meta: ...`` can be resolved —
    ``B``'s base ``A`` isn't in the iommi graph, but its iommi ancestor
    (``Table``) is. We use this map when scanning ``class Meta`` so the
    Meta is validated against the right refinable surface.
    """
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        base = _resolve_iommi_base(node, imports, graph, out)
        if base is not None:
            out[node.name] = base
    return out


def _resolve_iommi_base(
    cls_def: ast.ClassDef,
    imports: dict[str, str],
    graph: IommiGraph,
    user_iommi_subclasses: dict[str, str],
) -> str | None:
    """First base of *cls_def* that resolves to a class in the iommi graph.

    Recognised forms: direct names (``Table``), module-qualified
    attributes (``iommi.Table``), and references to another user class
    already known to ultimately extend an iommi class.
    """
    for base in cls_def.bases:
        qn = _resolve_callee(base, imports)
        if qn is None:
            continue
        cls = graph.get(qn)
        if cls is not None:
            return cls.qualname
        simple = qn.rsplit(".", 1)[-1]
        cls = graph.lookup_simple(simple)
        if cls is not None:
            return cls.qualname
        if simple in user_iommi_subclasses:
            return user_iommi_subclasses[simple]
        if qn in user_iommi_subclasses:
            return user_iommi_subclasses[qn]
    return None


def _find_meta_class(cls_def: ast.ClassDef) -> ast.ClassDef | None:
    for stmt in cls_def.body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == "Meta":
            return stmt
    return None


def _meta_assignment_targets(stmt: ast.stmt) -> list[ast.Name]:
    """Names assigned by *stmt* inside a Meta body.

    Only plain ``foo = ...`` and annotated ``foo: T = ...`` count. Tuple
    targets, attribute targets, and assignments-without-name-targets are
    not iommi refinable names and are skipped silently.
    """
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return [stmt.target]
    if isinstance(stmt, ast.Assign):
        return [t for t in stmt.targets if isinstance(t, ast.Name)]
    return []


def _validate_callable_value(kw: ast.keyword) -> Diagnostic | None:
    """Flag string-Constant values at known callable-expecting refinables.

    Iommi receives these as functions and calls them — passing a string
    silently breaks the view at request time. We only flag literal
    strings; Name / Attribute / Call / Lambda values pass through (ty
    catches undefined names; signature validation is a future problem).
    """
    value = kw.value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return None
    if value.lineno is None or value.col_offset is None:
        return None
    line0 = value.lineno - 1
    col_start = value.col_offset
    end_line = (value.end_lineno or value.lineno) - 1
    end_col = value.end_col_offset or (col_start + len(value.value) + 2)
    return {
        "code": _IOMMI_CALLABLE_CODE,
        "message": (
            f"{kw.arg!r} expects a callable; got a string literal — "
            f"did you mean to drop the quotes?"
        ),
        "range": {
            "start": {"line": line0, "character": col_start},
            "end": {"line": end_line, "character": end_col},
        },
        "severity": 2,
        "source": _IOMMI_DIAG_SOURCE,
    }


def _problem_at_name(
    name_node: ast.Name, chain: list[str], problem: Problem,
) -> Diagnostic:
    """Pin a diagnostic to the bad chain segment within an assignment target."""
    line0 = (name_node.lineno - 1) if name_node.lineno else 0
    name_col = name_node.col_offset
    sep = "__"
    seg_offset = 0
    for i, seg in enumerate(chain):
        if i == problem.segment_index:
            break
        seg_offset += len(seg) + len(sep)
    col_start = name_col + seg_offset
    col_end = col_start + len(problem.bad_segment)
    return _make_diagnostic(
        line0, col_start, col_end, _format_message(problem), problem,
    )


def _problem_to_diagnostic(
    source: str, kw: ast.keyword, chain: list[str], problem: Problem
) -> Diagnostic | None:
    """Pin the diagnostic to the specific bad segment within the kwarg name."""
    if kw.arg is None or kw.value is None:
        return None
    # ast on a keyword: `foo__bar=value`. The keyword name doesn't have
    # its own range in `ast` (Python doesn't track it precisely), so we
    # locate it by searching the source line.
    arg_name = kw.arg
    line0 = (kw.value.lineno - 1) if kw.value.lineno else 0
    line_text = source.splitlines()[line0] if line0 < len(source.splitlines()) else ""
    # Find the kwarg name on this line. Defensive against multi-line expressions.
    name_col = line_text.find(arg_name)
    if name_col == -1:
        # Multi-line expression: best-effort fall back to the value's range.
        col_start = (kw.value.col_offset or 0)
        col_end = col_start + 1
        return _make_diagnostic(
            line0,
            col_start,
            col_end,
            f"unknown iommi refinable {problem.bad_segment!r} on {problem.on_class}",
            problem,
        )

    # Compute the offset of the bad segment within `arg_name`.
    sep = "__"
    seg_offset = 0
    for i, seg in enumerate(chain):
        if i == problem.segment_index:
            break
        seg_offset += len(seg) + len(sep)

    col_start = name_col + seg_offset
    col_end = col_start + len(problem.bad_segment)

    return _make_diagnostic(
        line0,
        col_start,
        col_end,
        _format_message(problem),
        problem,
    )


def _format_message(problem: Problem) -> str:
    if problem.outcome == "unknown_refinable":
        msg = (
            f"unknown iommi refinable {problem.bad_segment!r} on "
            f"{problem.on_class}"
        )
        if problem.available:
            hint = ", ".join(problem.available[:8])
            if len(problem.available) > 8:
                hint += ", …"
            msg += f"  (available: {hint})"
        return msg
    if problem.outcome == "trailing_segments_after_leaf":
        return (
            f"refinable chain extends past a leaf at {problem.bad_segment!r}; "
            "the previous segment maps to a scalar/HTML attribute"
        )
    return f"invalid iommi refinable chain at {problem.bad_segment!r}"


def _offset_from_lsp_position(text: str, line: int, character: int) -> int:
    """Convert LSP ``{line, character}`` to a Python ``str`` offset.

    LSP characters are UTF-16 code units; non-BMP code points count as
    two. For ASCII source this collapses to straight character indexing.
    """
    offset = 0
    cur_line = 0
    n = len(text)
    while offset < n and cur_line < line:
        if text[offset] == "\n":
            cur_line += 1
        offset += 1
    char_units = 0
    while offset < n and char_units < character:
        ch = text[offset]
        if ch == "\n":
            break
        char_units += 2 if ord(ch) > 0xFFFF else 1
        offset += 1
    return offset


def _is_call_arg_position(source: str, partial_start: int) -> bool:
    """True if *partial_start* sits where a positional or kwarg name can
    begin — immediately after ``(`` or ``,`` with arbitrary whitespace.

    Skips the buffer ast.parse for cursors that are clearly outside any
    call (top-level identifiers, attribute access, dict keys, etc.).
    The single false-negative case is when the preceding token is a
    comment, which we don't try to scan past.
    """
    i = partial_start - 1
    while i >= 0 and source[i].isspace():
        i -= 1
    if i < 0:
        return False
    return source[i] in "(,"


def _is_meta_assignment_position(source: str, partial_start: int) -> bool:
    """True if *partial_start* sits at the start of a statement line
    (only whitespace separates it from the preceding newline) and the
    file contains a ``class Meta:`` declaration above the cursor.

    Cheap heuristic — the AST pass downstream confirms we're actually
    inside a Meta body. Without the substring guard we'd pay the parse
    cost for every top-level identifier in non-iommi files.
    """
    i = partial_start - 1
    while i >= 0 and source[i] in " \t":
        i -= 1
    if i >= 0 and source[i] != "\n":
        return False
    return "class Meta:" in source[:partial_start]


def _close_brackets(src: str) -> str:
    """Return the closing tokens needed to balance *src*. String-aware."""
    stack: list[str] = []
    pair = {"(": ")", "[": "]", "{": "}"}
    in_string: str | None = None
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        if in_string is not None:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == in_string:
                in_string = None
            i += 1
            continue
        if ch in '"\'':
            in_string = ch
        elif ch in "([{":
            stack.append(pair[ch])
        elif ch in ")]}":
            if stack and stack[-1] == ch:
                stack.pop()
        elif ch == "#":
            j = src.find("\n", i)
            i = n if j == -1 else j
            continue
        i += 1
    return "".join(reversed(stack))


@dataclass(frozen=True)
class _StringCtx:
    quote: str   # "'" or '"'
    start: int   # offset of the opening quote in source


def _string_state_at(source: str, offset: int) -> _StringCtx | None:
    """Return the open single-line string at *offset*, or None.

    Skips past closed triple-quoted spans (module docstrings) so they
    don't poison the state for code below.
    """
    i = 0
    n = min(offset, len(source))
    in_string: str | None = None
    string_start = -1
    while i < n:
        ch = source[i]
        if in_string is not None:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == in_string:
                in_string = None
                string_start = -1
            elif ch == "\n":
                # Single-quoted strings can't span a newline — treat as
                # broken and bail; the normal path will return empty.
                return None
            i += 1
            continue
        if ch in '"\'':
            if (
                i + 2 < n
                and source[i + 1] == ch
                and source[i + 2] == ch
            ):
                closing = source.find(ch * 3, i + 3, n)
                if closing == -1:
                    # Triple-quote doesn't close before the cursor — the
                    # cursor is inside a multi-line string, not eligible
                    # for single-line completion.
                    return None
                i = closing + 3
                continue
            in_string = ch
            string_start = i
        elif ch == "#":
            j = source.find("\n", i)
            i = n if j == -1 else j
            continue
        i += 1
    if in_string is None or string_start < 0:
        return None
    return _StringCtx(quote=in_string, start=string_start)


def _find_style_string(
    tree: ast.AST, str_start_offset: int,
) -> tuple[ast.Call, ast.Constant] | None:
    """Locate a string at *str_start_offset* that's the value of a
    ``style=`` kwarg. The enclosing callee isn't checked — iommi style
    names commonly appear on ``Style(...)``, ``Page(...)``,
    ``Table(...)`` and friends; over-offering across non-iommi calls
    just means an extra completion list the user can ignore.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "style":
                continue
            v = kw.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                if _node_offset_matches(v, str_start_offset, tree):
                    return node, v
    return None


def _find_auto_include_string(
    tree: ast.AST, str_start_offset: int,
) -> tuple[ast.Call, str, ast.Constant] | None:
    """Locate the (call, kwarg-name, string-node) at *str_start_offset*.

    The kwarg must be ``auto__include`` or ``auto__exclude`` and its
    value must be a ``List`` whose elements include a string ``Constant``
    starting at *str_start_offset*. Returns ``None`` if no such tuple
    exists in *tree*.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg not in ("auto__include", "auto__exclude"):
                continue
            if not isinstance(kw.value, ast.List):
                continue
            for elt in kw.value.elts:
                if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
                    continue
                # ast tracks line/col; we computed offset, so compare via
                # a synthetic offset derived from line/col is fragile —
                # but col_offset of a string Constant points at the
                # opening quote, and we know our str_start matches that.
                if _node_offset_matches(elt, str_start_offset, tree):
                    return node, kw.arg, elt
    return None


def _node_offset_matches(
    node: ast.Constant, target_offset: int, tree: ast.AST,
) -> bool:
    """Compare a Constant's source offset to *target_offset*.

    We don't keep a source-line index, so we lean on the fact that
    ast records ``lineno``/``col_offset`` (1-based line, 0-based col).
    The caller passes us the target_offset *in source units*; we
    convert the node's position to one offset using the original
    source via a one-shot helper attached as ``_source`` on *tree*.
    """
    src = getattr(tree, "_iommi_source", None)
    if not isinstance(src, str):
        return False
    line = node.lineno - 1
    col = node.col_offset
    node_offset = _line_col_to_offset(src, line, col)
    return node_offset == target_offset


def _line_col_to_offset(text: str, line: int, col: int) -> int:
    """Convert (line, col) — both 0-based — to a byte offset."""
    offset = 0
    n = len(text)
    cur = 0
    while offset < n and cur < line:
        if text[offset] == "\n":
            cur += 1
        offset += 1
    return offset + col


def _find_marker_call(tree: ast.AST, marker: str) -> ast.Call | None:
    """Return the smallest ``Call`` that has *marker* as a keyword name."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == marker:
                return node
    return None


def _find_marker_meta_owner(tree: ast.AST, marker: str) -> ast.ClassDef | None:
    """Outer class whose ``Meta`` body assigns to *marker*. Returns the
    outer class (not the ``Meta`` itself) so the caller can resolve its
    iommi base. ``None`` if the marker landed outside any Meta body
    (e.g. at module scope, inside a function, or in a class without a
    ``Meta:`` nested class)."""
    for outer in ast.walk(tree):
        if not isinstance(outer, ast.ClassDef):
            continue
        meta = _find_meta_class(outer)
        if meta is None:
            continue
        for stmt in meta.body:
            for target in _meta_assignment_targets(stmt):
                if target.id == marker:
                    return outer
    return None


_CONTAINER_KINDS = frozenset({
    "members", "class_ref", "traditional_class",
    "namespace", "open_namespace", "html_attrs",
})


# Canonical sub-keys of iommi's ``auto`` namespace. Reflected ``auto``
# refinables are usually ``open_namespace`` (the default ``Namespace()``
# is empty until the user sets ``auto__model=…``), which means the
# walker can't enumerate sub-keys from the graph alone. We treat
# ``auto`` as a synthetic namespace with these well-known keys so
# ``Table(auto__mo`` reliably suggests ``auto__model`` and ty's
# stray variable completions stay suppressed via exclusivity.
_AUTO_NAME = "auto"
_AUTO_KNOWN_KEYS: tuple[str, ...] = (
    "model", "rows", "instance", "include", "exclude",
)


# Well-known iommi classes that accept ``auto__model`` and have a members
# refinable. Used to synthesise a stub when the user hasn't run
# ``iommi_lsp graph build`` yet — the canonical shape is stable across
# iommi releases, so we can offer ``auto__`` + members completion
# without graph data.
_IOMMI_AUTO_BINDABLE_CLASSES: dict[str, str] = {
    "Table": "columns",
    "Form": "fields",
    "Query": "filters",
    "Page": "parts",
}


def _synthetic_auto_namespace() -> Refinable:
    return Refinable(
        name=_AUTO_NAME,
        kind="namespace",
        known_keys=list(_AUTO_KNOWN_KEYS),
    )


def _synthesize_iommi_class(qualname: str) -> IommiClass | None:
    """Build a stub ``IommiClass`` for a known iommi class.

    Returns ``None`` when the simple class name isn't in our hard-coded
    set — we deliberately don't synthesise on arbitrary Capital-Case
    callables, since that would mis-claim non-iommi calls (e.g. a
    user-defined ``MyService(...)`` would suddenly suggest ``auto__``).
    """
    simple = qualname.rsplit(".", 1)[-1]
    members_name = _IOMMI_AUTO_BINDABLE_CLASSES.get(simple)
    if members_name is None:
        return None
    return IommiClass(
        qualname=qualname,
        bases=[],
        refinables={
            members_name: Refinable(name=members_name, kind="members"),
            _AUTO_NAME: _synthetic_auto_namespace(),
        },
    )


def _is_container(kind: str) -> bool:
    return kind in _CONTAINER_KINDS


def _complete_top_level(
    graph: IommiGraph, cls: IommiClass, partial: str
) -> list[dict]:
    """Refinable items at the top level of a `Class(...)` call."""
    names = set(_all_refinables(graph, cls))
    # Synthesize `auto` for classes that can plausibly auto-bind a model
    # — i.e. they expose a ``members`` refinable (Table.columns,
    # Form.fields, Query.filters, ...). Reflection sometimes misses
    # `auto` because the default ``Namespace()`` is empty, but the
    # keyword is part of iommi's public API regardless.
    if _AUTO_NAME not in names and _has_members_refinable(graph, cls):
        names.add(_AUTO_NAME)

    out: list[dict] = []
    for name in sorted(names):
        if partial and not name.startswith(partial):
            continue
        ref = _resolve_refinable(graph, cls, name)
        if ref is None and name == _AUTO_NAME:
            ref = _synthetic_auto_namespace()
        out.append(_refinable_item(name, ref, cls.qualname))
    return out


def _has_members_refinable(graph: IommiGraph, cls: IommiClass) -> bool:
    """Does *cls* (or any base) expose at least one ``members`` refinable?

    Used as a cheap proxy for "this class plausibly supports
    ``auto__model``" — only members-bearing classes (Table, Form, Query,
    Page) accept auto-binding.
    """
    for name in _all_refinables(graph, cls):
        ref = _resolve_refinable(graph, cls, name)
        if ref is not None and ref.kind == "members":
            return True
    return False


def _complete_after_chain(
    graph: IommiGraph,
    start_cls: IommiClass,
    chain: list[str],
    partial: str,
    *,
    prefix: str,
    auto_model: "ModelInfo | None" = None,
) -> list[dict] | None:
    """Walk *chain* from *start_cls* and return completion items.

    Returns ``None`` when the chain hits something we can't enumerate
    (open namespace, unknown segment, member slot without an
    ``auto__model`` to back it). An empty list means the chain
    terminated at a leaf.
    """
    return _enumerate_from_class(
        graph, start_cls, chain, 0, partial, prefix, auto_model=auto_model,
    )


def _enumerate_from_class(
    graph: IommiGraph,
    cls: IommiClass,
    chain: list[str],
    i: int,
    partial: str,
    prefix: str,
    *,
    auto_model: "ModelInfo | None",
) -> list[dict] | None:
    if i >= len(chain):
        out: list[dict] = []
        for name in sorted(_all_refinables(graph, cls)):
            if partial and not name.startswith(partial):
                continue
            ref = _resolve_refinable(graph, cls, name)
            out.append(_refinable_item(name, ref, cls.qualname, prefix=prefix))
        return out

    seg = chain[i]
    if seg == _AUTO_NAME:
        # `auto` is iommi's canonical model-binding namespace — its
        # default Namespace is empty, so the graph reflects it as
        # ``open_namespace``. Override with synthetic known_keys so we
        # can offer ``auto__model`` / ``auto__rows`` / ``auto__include``
        # etc. regardless of what the graph says.
        return _enumerate_from_refinable(
            graph, _synthetic_auto_namespace(), chain, i + 1, partial, prefix,
            parent=cls, auto_model=auto_model,
        )
    ref = _resolve_refinable(graph, cls, seg)
    if ref is None:
        if seg == _ATTRS_NAME:
            return _enumerate_from_refinable(
                graph, _synthetic_html_attrs(), chain, i + 1, partial, prefix,
                parent=cls, auto_model=auto_model,
            )
        return None
    return _enumerate_from_refinable(
        graph, ref, chain, i + 1, partial, prefix,
        parent=cls, auto_model=auto_model,
    )


def _enumerate_from_refinable(
    graph: IommiGraph,
    ref: Refinable,
    chain: list[str],
    j: int,
    partial: str,
    prefix: str,
    *,
    parent: IommiClass,
    auto_model: "ModelInfo | None",
) -> list[dict] | None:
    remaining = chain[j:]
    kind = ref.kind

    if kind in ("scalar", "evaluated_scalar"):
        return []   # leaf

    if kind == "open_namespace":
        return None

    if kind == "namespace":
        if not remaining:
            return [
                _namespace_key_item(k, parent.qualname, prefix=prefix)
                for k in ref.known_keys
                if not partial or k.startswith(partial)
            ]
        next_seg = remaining[0]
        if ref.known_keys and next_seg not in ref.known_keys:
            return None
        if next_seg == _ATTRS_NAME:
            return _enumerate_from_refinable(
                graph, _synthetic_html_attrs(), chain, j + 1, partial, prefix,
                parent=parent, auto_model=auto_model,
            )
        return None

    if kind == "html_attrs":
        if not remaining:
            return [
                _html_attrs_item(name, parent.qualname, prefix=prefix)
                for name in ("class", "style")
                if not partial or name.startswith(partial)
            ]
        head = remaining[0]
        if head not in ref.sub_specials:
            return []   # arbitrary attr is itself the leaf
        return None

    if kind == "class_ref":
        if ref.target is None:
            return None
        target = graph.get(ref.target)
        if target is None:
            return None
        return _enumerate_from_class(
            graph, target, chain, j, partial, prefix, auto_model=auto_model,
        )

    if kind == "traditional_class":
        if ref.target is None:
            return None
        target = graph.get(ref.target)
        if target is None or not target.init_members:
            return None
        if not remaining:
            return [
                _init_member_item(name, target.qualname, prefix=prefix)
                for name in target.init_members
                if not partial or name.startswith(partial)
            ]
        # Any segment past a traditional class's leaf init_member is invalid
        # — drop completions so the user sees nothing rather than fake hits.
        return []

    if kind == "members":
        if not remaining:
            # User-chosen member-name slot. If iommi auto-binds this
            # members refinable to a Django model (``Table(auto__model=
            # User, columns__...)``), the names ARE that model's fields.
            if auto_model is not None:
                return _model_field_items(
                    auto_model, parent.qualname, prefix=prefix, partial=partial,
                )
            return None
        if ref.member_class is None:
            return None
        target = graph.get(ref.member_class)
        if target is None:
            return None
        return _enumerate_from_class(
            graph, target, chain, j + 1, partial, prefix, auto_model=auto_model,
        )

    return None


def _refinable_detail(ref: Refinable | None) -> str:
    if ref is None:
        return ""
    return f"iommi refinable ({ref.kind})"


def _refinable_item(
    name: str, ref: Refinable | None, cls_qualname: str, *, prefix: str = "",
) -> dict:
    """Build a completion item for a refinable. Container refinables get
    a trailing ``__`` so accepting the completion lands the cursor at
    the next chain position; scalars get ``=`` so the cursor is ready
    for a value.
    """
    full = f"{prefix}{name}"
    is_container = ref is not None and _is_container(ref.kind)
    if is_container:
        label = f"{full}__"
        insert = f"{full}__"
    else:
        label = full
        insert = f"{full}="
    return {
        "label": label,
        "kind": 5,
        "insertText": insert,
        "detail": _refinable_detail(ref),
        "data": {"source": "iommi_lsp.iommi-kwarg", "class": cls_qualname},
    }


def _namespace_key_item(key: str, cls_qualname: str, *, prefix: str) -> dict:
    """A key inside a ``namespace`` refinable. We don't have type info
    for these — the only one we *know* is a container is ``attrs``
    (always html_attrs); the rest default to scalar (``=``)."""
    full = f"{prefix}{key}"
    if key == _ATTRS_NAME:
        return {
            "label": f"{full}__",
            "kind": 5,
            "insertText": f"{full}__",
            "detail": "html attrs",
            "data": {"source": "iommi_lsp.iommi-kwarg", "class": cls_qualname},
        }
    return {
        "label": full,
        "kind": 5,
        "insertText": f"{full}=",
        "detail": "namespace key",
        "data": {"source": "iommi_lsp.iommi-kwarg", "class": cls_qualname},
    }


def _init_member_item(name: str, cls_qualname: str, *, prefix: str) -> dict:
    """Init-member completion item — a leaf scalar; suffix with ``=``."""
    full = f"{prefix}{name}"
    return {
        "label": full,
        "kind": 5,
        "insertText": f"{full}=",
        "detail": f"{cls_qualname}.__init__ attribute",
        "data": {"source": "iommi_lsp.iommi-kwarg", "class": cls_qualname},
    }


def _html_attrs_item(name: str, cls_qualname: str, *, prefix: str) -> dict:
    """``class`` / ``style`` — dict-keyed, drill in."""
    full = f"{prefix}{name}"
    detail = "css class dict" if name == "class" else "css style dict"
    return {
        "label": f"{full}__",
        "kind": 5,
        "insertText": f"{full}__",
        "detail": detail,
        "data": {"source": "iommi_lsp.iommi-kwarg", "class": cls_qualname},
    }


def _model_field_items(
    model: "ModelInfo", cls_qualname: str, *, prefix: str, partial: str,
) -> list[dict]:
    """Fields of an ``auto__model``-bound Django model as member-name
    completions. Each entry drills in with ``__`` so the user can keep
    configuring the auto-generated column/field (the canonical iommi
    pattern: ``columns__username__display_name='...'``)."""
    out: list[dict] = []
    for name in sorted(model.fields):
        if partial and not name.startswith(partial):
            continue
        fi = model.fields[name]
        full = f"{prefix}{name}"
        out.append({
            "label": f"{full}__",
            "kind": 5,
            "insertText": f"{full}__",
            "detail": f"auto-bound {fi.field_type} on {model.qualname}",
            "data": {
                "source": "iommi_lsp.iommi-kwarg-auto",
                "class": cls_qualname,
                "model": model.qualname,
            },
        })
    return out


# ---------------------------------------------------------------------------
# auto__model resolution against the Django index
# ---------------------------------------------------------------------------


def _resolve_model_from_name(value: ast.AST, index) -> "ModelInfo | None":
    """Bare ``Name`` referring to a Django model (``auto__model=User``)."""
    if isinstance(value, ast.Name):
        return index.lookup(value.id)
    if isinstance(value, ast.Attribute):
        # ``auto__model=models.User`` / ``auto__model=myapp.models.User``.
        return index.lookup(value.attr)
    return None


def _resolve_model_from_manager_chain(value: ast.AST, index) -> "ModelInfo | None":
    """``Model.objects.<method>(...)`` chains — for auto__rows / instance."""
    cur = value
    while isinstance(cur, ast.Call):
        if not isinstance(cur.func, ast.Attribute):
            return None
        cur = cur.func.value
    if isinstance(cur, ast.Attribute):
        if cur.attr not in ("objects", "_default_manager", "_base_manager"):
            return None
        owner = cur.value
        if isinstance(owner, ast.Name):
            return index.lookup(owner.id)
        if isinstance(owner, ast.Attribute):
            return index.lookup(owner.attr)
    return None


def _make_diagnostic(
    line: int, col_start: int, col_end: int, message: str, problem: Problem
) -> Diagnostic:
    return {
        "code": _IOMMI_DIAG_CODE,
        "message": message,
        "range": {
            "start": {"line": line, "character": col_start},
            "end": {"line": line, "character": col_end},
        },
        "severity": 2,   # warning — bias toward false negatives
        "source": _IOMMI_DIAG_SOURCE,
        "data": {
            "outcome": problem.outcome,
            "on_class": problem.on_class,
            "available": list(problem.available),
        },
    }
