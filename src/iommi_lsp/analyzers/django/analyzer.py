"""Django analyzer: drops ``unresolved-attribute`` diagnostics whose
target is a Django metaclass-injected attribute on a recognised model.

For v1 we recognise the receiver type via two cheap heuristics
(``DESIGN.md`` §6.3):

* (a) **Syntactic match.** Bare ``Name`` whose simple identifier matches
  a known model class — e.g. ``User.objects``.
* (b) **Local flow.** Same-function assignments where the RHS is
  ``Model(...)`` or ``Model.objects.<query>(...)``  — e.g.
  ``user = User(...); user.pk``. Only the most recent assignment wins.

Anything outside these cases is forwarded unchanged. The bias is
explicitly toward false negatives: we'd rather leak some noise than
suppress a real bug.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from typing import TYPE_CHECKING

from ... import log
from ..base import Analyzer, CompletionResult, Diagnostic
from . import lookup_walker
from .index import (
    DjangoIndex,
    ModelInfo,
    _FileScrape,
    assemble_index,
    collect_scrapes,
    update_scrapes,
)
from .magic import FK_LIKE_FIELD_NAMES, ORM_LOOKUP_NAMES, RELATION_FIELD_NAMES

if TYPE_CHECKING:
    from ...config import Config


_log = log.get("django.analyzer")


_QUERY_METHODS_RETURNING_INSTANCE = frozenset({
    "get", "first", "last", "earliest", "latest",
    "create", "get_or_create", "update_or_create",
})

# Manager methods that take ``field__lookup=...`` kwargs we want to validate.
# ``update``/``create`` only accept single-segment field names in real
# Django — the walker is permissive about ``__`` traversal which means
# we'd miss e.g. ``update(author__name='x')`` (invalid Django). Bias FN.
_LOOKUP_METHODS = frozenset({
    "filter", "exclude", "get", "get_or_create", "update_or_create",
    "update", "create",
})

# Methods whose positional args are field-path strings (``order_by``-style).
# Each string is a chain like ``"author__name"`` (with optional leading
# ``-`` for ``order_by`` descending; ``"?"`` for random).
_FIELD_PATH_METHODS = frozenset({
    "order_by", "values", "values_list", "only", "defer", "distinct",
    "select_related", "prefetch_related",
})

_MANAGER_NAMES = frozenset({"objects", "_default_manager", "_base_manager"})

# Kwargs that some manager methods accept which are NOT field names
# (e.g. `defaults={...}` to ``get_or_create``). Skipping them keeps the
# scanner from raising spurious "unknown field" diagnostics.
_METHOD_ONLY_KWARGS = frozenset({"defaults", "create_defaults"})

_ORM_LOOKUP_DIAG_CODE = "django-unknown-orm-lookup"
_ORM_LOOKUP_DIAG_SOURCE = "iommi_lsp"


@dataclass
class _ParsedFile:
    tree: ast.Module
    source: str


class DjangoAnalyzer:
    """Implements the :class:`Analyzer` Protocol."""

    name = "django"

    def __init__(
        self,
        workspace_root: Path,
        django_index: DjangoIndex | None = None,
        config: "Config | None" = None,
        text_provider: Callable[[str], str | None] | None = None,
    ) -> None:
        # Lazy import — config.py pulls in this package via magic.py and we
        # need to break the cycle.
        from ...config import DEFAULT as DEFAULT_CONFIG

        self.workspace_root = workspace_root
        self.django_index: DjangoIndex = django_index or DjangoIndex()
        self.config: "Config" = config or DEFAULT_CONFIG
        self._text_provider = text_provider
        self._cache: dict[str, _ParsedFile] = {}
        self._scrapes: dict[Path, _FileScrape] = {}

    # -- Analyzer protocol ----------------------------------------------------

    async def index(self, workspace_root: Path) -> None:
        from ...config import load as load_config

        self.workspace_root = workspace_root
        self.config = load_config(workspace_root)
        self._scrapes = collect_scrapes(workspace_root)
        self.django_index = assemble_index(workspace_root, self._scrapes)
        self._cache.clear()

    async def on_file_changed(self, uri: str) -> None:
        self._cache.pop(uri, None)
        path = _uri_to_path(uri)
        if path is None:
            return
        # Incremental: only re-parse the changed file, then re-run
        # classification + reverse-relation computation against the
        # cached scrape map. ~milliseconds even on large workspaces.
        update_scrapes(self.workspace_root, self._scrapes, path)
        self.django_index = assemble_index(self.workspace_root, self._scrapes)

    # -- internals ------------------------------------------------------------

    def _evaluate(self, uri: str, diagnostic: Diagnostic) -> bool:
        path = _uri_to_path(uri)
        if path is None:
            return False
        parsed = self._parse(uri, path)
        if parsed is None:
            return False

        attr_node = _find_attribute_at(parsed.tree, diagnostic.get("range") or {})
        if attr_node is None:
            return False
        attr_name = attr_node.attr
        receiver = attr_node.value

        model = self._resolve_receiver_model(receiver, parsed.tree)
        if model is None:
            return False

        return self._attr_is_magic(model, attr_name)

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

    def _resolve_receiver_model(
        self, receiver: ast.AST, tree: ast.Module
    ) -> ModelInfo | None:
        # (a) Syntactic match: bare Name -> class lookup by simple name.
        if isinstance(receiver, ast.Name):
            model = self.django_index.lookup(receiver.id)
            if model is not None:
                return model
            # (b) Local flow: search enclosing scope for an assignment.
            return self._resolve_local_variable(receiver.id, receiver, tree)
        return None

    def _resolve_local_variable(
        self, var_name: str, use_site: ast.AST, tree: ast.Module
    ) -> ModelInfo | None:
        scope = _enclosing_function(tree, use_site)
        if scope is None:
            scope = tree
        # Iterate assignments preceding the use site; last one wins.
        last_match: ModelInfo | None = None
        use_pos = (getattr(use_site, "lineno", 0), getattr(use_site, "col_offset", 0))
        for stmt in ast.walk(scope):
            if not isinstance(stmt, ast.Assign):
                continue
            if (stmt.lineno, stmt.col_offset) >= use_pos:
                continue
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name) and tgt.id == var_name:
                    inferred = self._infer_call_result_model(stmt.value)
                    if inferred is not None:
                        last_match = inferred
        return last_match

    def _infer_call_result_model(self, value: ast.AST) -> ModelInfo | None:
        """Recognise ``Model(...)`` and ``Model.objects.<method>(...)``."""
        # Model(...) — direct instantiation.
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            return self.django_index.lookup(value.func.id)
        # Model.objects.<method>(...) — manager call.
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            method = value.func.attr
            if method not in _QUERY_METHODS_RETURNING_INSTANCE:
                return None
            mgr = value.func.value
            if (
                isinstance(mgr, ast.Attribute)
                and mgr.attr in {"objects", "_default_manager", "_base_manager"}
                and isinstance(mgr.value, ast.Name)
            ):
                return self.django_index.lookup(mgr.value.id)
        return None

    def _attr_is_magic(self, model: ModelInfo, attr_name: str) -> bool:
        cfg = self.config

        for group in ("manager", "meta", "exception"):
            if cfg.is_rule_enabled(group) and attr_name in cfg.merged_static_attrs(group):
                return True

        if cfg.is_rule_enabled("pk") and attr_name in cfg.merged_static_attrs("pk"):
            # Special-case `id`: only present implicitly when no explicit PK.
            if attr_name == "id" and not model.implicit_id:
                return False
            return True

        if cfg.is_rule_enabled("fk_id") and attr_name in model.fk_id_accessors:
            return True

        if cfg.is_rule_enabled("reverse") and attr_name in self.django_index.reverse_attrs(model.qualname):
            return True

        return False

    def is_false_positive(self, uri: str, diagnostic: Diagnostic) -> bool:  # type: ignore[override]
        if not self.config.enabled:
            return False
        if (
            _is_unused_request(diagnostic)
            and self.config.is_rule_enabled("unused_request_param")
        ):
            try:
                return self._is_first_request_param(uri, diagnostic)
            except Exception:
                _log.exception("unused-request check crashed; keeping the diagnostic")
                return False
        if not _is_unresolved_attribute(diagnostic):
            return False
        try:
            return self._evaluate(uri, diagnostic)
        except Exception:
            _log.exception("analyzer crashed; keeping the diagnostic")
            return False

    def _is_first_request_param(self, uri: str, diagnostic: Diagnostic) -> bool:
        path = _uri_to_path(uri)
        if path is None:
            return False
        parsed = self._parse(uri, path)
        if parsed is None:
            return False
        rng = diagnostic.get("range") or {}
        start = rng.get("start") or {}
        # LSP positions are 0-indexed; AST line numbers are 1-indexed.
        line_no = int(start.get("line", 0)) + 1
        col = int(start.get("character", 0))
        for node in ast.walk(parsed.tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = list(node.args.posonlyargs) + list(node.args.args)
            # Skip self/cls so class-based views (`def get(self, request, ...)`)
            # are still treated as having `request` first.
            if params and params[0].arg in ("self", "cls"):
                params = params[1:]
            if not params:
                continue
            first = params[0]
            if first.arg != "request":
                continue
            if first.lineno == line_no and first.col_offset == col:
                return True
        return False

    def additional_diagnostics(self, uri: str) -> list[Diagnostic]:
        if not self.config.is_rule_enabled("orm_lookup"):
            return []
        if not self.django_index.models:
            return []
        path = _uri_to_path(uri)
        if path is None:
            return []
        parsed = self._parse(uri, path)
        if parsed is None:
            return []
        try:
            return list(self._scan_lookups(parsed))
        except Exception:
            _log.exception("orm-lookup scanner crashed; emitting nothing")
            return []

    def completions(self, uri: str, position: dict) -> CompletionResult:
        """Return LSP ``CompletionItem`` dicts for an ORM-lookup kwarg.

        Triggers when the cursor sits inside ``Model.objects.<method>(...)``
        at a position that could be the start of a kwarg name (the same
        method set we validate, ``filter``/``exclude``/``get``/``update``/
        ``create``/``get_or_create``/``update_or_create``). On a
        recognised position the result is *exclusive*: ty's name-of-any-
        variable completions are noise here, so the matchmaker drops
        them. On any uncertainty about the call site or the receiver
        model, we return an empty non-exclusive result and let ty's
        completions through.
        """
        empty = CompletionResult()
        if not self.config.is_rule_enabled("orm_lookup"):
            return empty
        if not self.django_index.models:
            return empty
        path = _uri_to_path(uri)
        if path is None:
            return empty
        source = self._source_for(uri, path)
        if source is None:
            return empty
        try:
            return self._scan_completions(source, position)
        except Exception:
            _log.exception("completion scanner crashed; emitting nothing")
            return empty

    def _scan_completions(self, source: str, position: dict) -> CompletionResult:
        empty = CompletionResult()
        line = int(position.get("line", 0))
        character = int(position.get("character", 0))
        offset = _offset_from_lsp_position(source, line, character)
        if offset > len(source):
            return empty

        # Walk back over identifier chars at the cursor to find the partial
        # kwarg name the user has typed so far (may be empty).
        partial_start = offset
        while partial_start > 0 and (
            source[partial_start - 1].isalnum()
            or source[partial_start - 1] == "_"
        ):
            partial_start -= 1
        partial = source[partial_start:offset]

        # Patch the source so it parses: replace everything from the
        # partial onward with `<marker>=None`, then append closes to
        # balance any open brackets above the cursor. This turns an
        # in-progress `filter(em` into `filter(__marker__=None)` which
        # ast can chew on; the marker keyword is what we find later.
        marker = "__iommi_lsp_completion_marker__"
        head = source[:partial_start]
        inserted = marker + "=None"
        closes = _close_brackets(head + inserted)
        patched = head + inserted + closes

        try:
            tree = ast.parse(patched)
        except SyntaxError:
            return empty

        marker_call = _find_marker_call(tree, marker)
        if marker_call is None:
            return empty
        if not isinstance(marker_call.func, ast.Attribute):
            return empty
        method = marker_call.func.attr
        if method not in _LOOKUP_METHODS:
            return empty

        model = self._root_manager_model(marker_call.func.value, tree)
        if model is None:
            # We recognised the call shape but can't say which model —
            # don't suppress ty here, the user might know better.
            return empty

        # `foreign_key__name` — walk the chain so completions reflect the
        # target model, not the receiver. The last `__`-segment is the
        # in-progress identifier; everything before it is the chain.
        if "__" in partial:
            head, _, suffix = partial.rpartition("__")
            chain = lookup_walker.split_chain(head)
            target = _walk_chain_for_completion(
                self.django_index, model, chain
            )
            if target is None:
                # Chain didn't resolve to something we can complete on
                # (unknown segment, or terminated at a non-relation leaf).
                # Still our position — bias toward exclusive empty.
                return CompletionResult(items=[], exclusive=True)
            items = list(_field_completion_items(
                self.django_index, target, suffix, prefix=head + "__"
            ))
        else:
            items = list(_field_completion_items(
                self.django_index, model, partial
            ))
        return CompletionResult(items=items, exclusive=True)

    def _scan_lookups(self, parsed: _ParsedFile):
        for node in ast.walk(parsed.tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            method = node.func.attr
            if method not in _LOOKUP_METHODS and method not in _FIELD_PATH_METHODS:
                continue
            model = self._root_manager_model(node.func.value, parsed.tree)
            if model is None:
                continue
            if method in _LOOKUP_METHODS:
                # `.filter(name__icontains=…)` — direct kwargs.
                yield from self._validate_kwargs(parsed, model, node.keywords)
                # `.filter(Q(a=1) | Q(b=2), …)` — kwargs inside Q expressions.
                for arg in node.args:
                    for q_kwargs in _iter_q_kwargs(arg):
                        yield from self._validate_kwargs(parsed, model, q_kwargs)
            if method in _FIELD_PATH_METHODS:
                yield from self._validate_field_path_args(
                    parsed, model, node.args, method
                )
            # F('field__path') anywhere in the call's args/kwargs.
            yield from self._validate_f_calls(model, node)

    def _validate_field_path_args(
        self,
        parsed: _ParsedFile,
        model: ModelInfo,
        args: list[ast.expr],
        method: str,
    ):
        for arg in args:
            if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                # F() / Prefetch() / variables — skip.
                continue
            raw = arg.value
            # `order_by('?')` — random ordering, not a field path.
            if method == "order_by" and raw == "?":
                continue
            leading = 0
            if method == "order_by" and raw.startswith("-"):
                leading = 1
            chain_str = raw[leading:]
            if not chain_str:
                continue
            chain = lookup_walker.split_chain(chain_str)
            result = lookup_walker.walk(self.django_index, model.qualname, chain)
            if isinstance(result, lookup_walker.Problem):
                diag = _string_problem_to_diagnostic(arg, chain, leading, result)
                if diag is not None:
                    yield diag

    def _validate_f_calls(self, model: ModelInfo, call: ast.Call):
        """Find F('field__path') anywhere in *call*'s arg/kwarg subtrees."""
        seen: set[int] = set()
        for sub in _iter_arg_subtrees(call):
            for fnode in ast.walk(sub):
                if not isinstance(fnode, ast.Call) or not _is_f_call(fnode):
                    continue
                key = id(fnode)
                if key in seen:
                    continue
                seen.add(key)
                if not fnode.args:
                    continue
                arg0 = fnode.args[0]
                if not isinstance(arg0, ast.Constant) or not isinstance(arg0.value, str):
                    continue
                chain = lookup_walker.split_chain(arg0.value)
                result = lookup_walker.walk(self.django_index, model.qualname, chain)
                if isinstance(result, lookup_walker.Problem):
                    diag = _string_problem_to_diagnostic(arg0, chain, 0, result)
                    if diag is not None:
                        yield diag

    def _validate_kwargs(
        self,
        parsed: _ParsedFile,
        model: ModelInfo,
        kwargs: list[ast.keyword],
    ):
        for kw in kwargs:
            if kw.arg is None:
                continue   # **kwargs splat
            if kw.arg in _METHOD_ONLY_KWARGS:
                continue
            chain = lookup_walker.split_chain(kw.arg)
            result = lookup_walker.walk(self.django_index, model.qualname, chain)
            if isinstance(result, lookup_walker.Problem):
                diag = _problem_to_diagnostic(parsed.source, kw, chain, result)
                if diag is not None:
                    yield diag

    def _root_manager_model(
        self,
        receiver: ast.AST,
        tree: ast.Module | None = None,
        visited: frozenset[str] = frozenset(),
    ) -> ModelInfo | None:
        """Walk back through chained calls until we hit a manager-rooted form.

        Recognises three rooted shapes:

        * ``Model.<manager>`` — the canonical case (``User.objects``).
        * ``<dotted>.Model.<manager>`` — module-qualified access
          (``myapp.models.User.objects``); the rightmost segment is used
          as a simple-name lookup, which the index naturally
          ambiguity-protects.
        * A bare local variable previously assigned from one of the
          above, possibly through queryset-returning chains
          (``qs = User.objects.all(); qs.filter(...)``).

        Returns the model if the leftmost receiver is recognised; ``None``
        otherwise (which means we don't validate that call). The
        ``visited`` set guards against self-referential assignment loops.
        """
        cur = receiver
        # Peel off any chain of method calls: each call's func is an
        # Attribute whose value is the previous receiver.
        while isinstance(cur, ast.Call):
            if not isinstance(cur.func, ast.Attribute):
                return None
            cur = cur.func.value

        # `<…>.<manager>` — direct or module-qualified.
        if isinstance(cur, ast.Attribute):
            if cur.attr not in _MANAGER_NAMES:
                return None
            owner = cur.value
            if isinstance(owner, ast.Name):
                return self.django_index.lookup(owner.id)
            if isinstance(owner, ast.Attribute):
                # `models.User.objects` / `app.models.User.objects`.
                return self.django_index.lookup(owner.attr)
            return None

        # Bare name — could be a local queryset variable.
        if isinstance(cur, ast.Name) and tree is not None:
            if cur.id in visited:
                return None
            return self._resolve_local_queryset_model(
                cur.id, cur, tree, visited | {cur.id}
            )

        return None

    def _resolve_local_queryset_model(
        self,
        var_name: str,
        use_site: ast.AST,
        tree: ast.Module,
        visited: frozenset[str],
    ) -> ModelInfo | None:
        """Resolve a local variable to the model its queryset is bound to.

        Walks same-function assignments preceding *use_site*; the most
        recent successfully-resolved RHS wins. Recursion is bounded by
        *visited* (see :meth:`_root_manager_model`).
        """
        scope = _enclosing_function(tree, use_site)
        if scope is None:
            scope = tree
        last_match: ModelInfo | None = None
        use_pos = (
            getattr(use_site, "lineno", 0),
            getattr(use_site, "col_offset", 0),
        )
        for stmt in ast.walk(scope):
            if not isinstance(stmt, ast.Assign):
                continue
            if (stmt.lineno, stmt.col_offset) >= use_pos:
                continue
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name) and tgt.id == var_name:
                    inferred = self._root_manager_model(
                        stmt.value, tree, visited
                    )
                    if inferred is not None:
                        last_match = inferred
        return last_match


# ---------------------------------------------------------------------------
# Helpers — kept module-level so they're easy to test in isolation later.
# ---------------------------------------------------------------------------


def _is_unresolved_attribute(diagnostic: Diagnostic) -> bool:
    code = diagnostic.get("code")
    if isinstance(code, str) and code == "unresolved-attribute":
        return True
    # Some clients normalize ``code`` as an int; ty uses strings, but stay safe.
    if isinstance(code, dict) and code.get("value") == "unresolved-attribute":
        return True
    return False


def _is_unused_request(diagnostic: Diagnostic) -> bool:
    """Match ty's hint ``\\`request\\` is unused``. ty emits this with no
    diagnostic code, so we sniff source + message text directly.
    """
    if diagnostic.get("source") != "ty":
        return False
    message = diagnostic.get("message")
    if not isinstance(message, str):
        return False
    return message.strip() == "`request` is unused"


def _uri_to_path(uri: str) -> Path | None:
    if not uri.startswith("file://"):
        return None
    parsed = urlparse(uri)
    return Path(unquote(parsed.path))


def _find_attribute_at(tree: ast.Module, range_: dict) -> ast.Attribute | None:
    """Find the smallest ``ast.Attribute`` node containing the LSP range."""
    start = range_.get("start") or {}
    end = range_.get("end") or {}
    s_line = int(start.get("line", 0)) + 1   # LSP is 0-indexed, AST is 1-indexed
    s_col = int(start.get("character", 0))
    e_line = int(end.get("line", s_line - 1)) + 1
    e_col = int(end.get("character", s_col))

    best: ast.Attribute | None = None
    best_size = (10**9, 10**9)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        nl = node.lineno
        nc = node.col_offset
        nel = node.end_lineno or nl
        nec = node.end_col_offset or nc
        # Node range must contain the diagnostic range.
        if (nl, nc) > (s_line, s_col):
            continue
        if (nel, nec) < (e_line, e_col):
            continue
        size = (nel - nl, nec - nc)
        if size < best_size:
            best = node
            best_size = size
    return best


def _is_f_call(call: ast.Call) -> bool:
    """Recognise ``F(...)`` and ``models.F(...)`` calls."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "F"
    if isinstance(func, ast.Attribute):
        return func.attr == "F"
    return False


def _iter_arg_subtrees(call: ast.Call):
    """Yield the AST subtrees of *call*'s positional + keyword args.

    Avoids the ``func`` subtree so chained-receiver calls aren't
    re-scanned (each chained call is reached on its own ``ast.walk``).
    """
    for a in call.args:
        yield a
    for kw in call.keywords:
        if kw.value is not None:
            yield kw.value


def _is_q_call(call: ast.Call) -> bool:
    """Recognise ``Q(...)`` and ``models.Q(...)`` / ``...Q(...)`` calls."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "Q"
    if isinstance(func, ast.Attribute):
        return func.attr == "Q"
    return False


def _iter_q_kwargs(node: ast.AST):
    """Yield the keyword lists of every Q(...) call reachable from *node*.

    Walks through ``|`` / ``&`` (BinOp) and ``~`` (UnaryOp) since Q
    expressions compose via boolean operators. Bare ``Q`` references
    (variables, attribute access without a call) are ignored — we don't
    follow data flow.
    """
    if isinstance(node, ast.Call):
        if _is_q_call(node):
            yield node.keywords
            # Q(Q(a=1), b=2) — nested Q in positional args.
            for sub in node.args:
                yield from _iter_q_kwargs(sub)
        return
    if isinstance(node, ast.BoolOp):
        for v in node.values:
            yield from _iter_q_kwargs(v)
        return
    if isinstance(node, ast.BinOp):
        yield from _iter_q_kwargs(node.left)
        yield from _iter_q_kwargs(node.right)
        return
    if isinstance(node, ast.UnaryOp):
        yield from _iter_q_kwargs(node.operand)
        return


def _problem_to_diagnostic(
    source: str,
    kw: ast.keyword,
    chain: list[str],
    problem: lookup_walker.Problem,
) -> Diagnostic | None:
    """Pin the diagnostic to the bad segment within the kwarg name."""
    if kw.arg is None or kw.value is None:
        return None
    arg_name = kw.arg
    line0 = (kw.value.lineno - 1) if kw.value.lineno else 0
    lines = source.splitlines()
    line_text = lines[line0] if 0 <= line0 < len(lines) else ""
    # Anchor on `arg_name=` so a kwarg name that also appears earlier as
    # a value (rare, but possible) doesn't mis-pin us.
    needle = f"{arg_name}="
    name_col = line_text.find(needle)
    if name_col == -1:
        name_col = line_text.find(arg_name)

    if name_col == -1:
        col_start = kw.value.col_offset or 0
        col_end = col_start + 1
        return _make_orm_diagnostic(
            line0, col_start, col_end, _format_orm_message(problem), problem
        )

    sep = "__"
    seg_offset = 0
    for i, seg in enumerate(chain):
        if i == problem.segment_index:
            break
        seg_offset += len(seg) + len(sep)

    col_start = name_col + seg_offset
    col_end = col_start + len(problem.bad_segment)
    return _make_orm_diagnostic(
        line0, col_start, col_end, _format_orm_message(problem), problem
    )


def _string_problem_to_diagnostic(
    arg: ast.Constant,
    chain: list[str],
    leading: int,
    problem: lookup_walker.Problem,
) -> Diagnostic | None:
    """Pin a diagnostic to the bad segment inside a string-literal field path.

    *leading* is the count of source characters consumed before the chain
    begins (e.g. ``1`` for ``order_by('-foo')`` to skip the ``-``).
    """
    if arg.lineno is None or arg.col_offset is None:
        return None
    line0 = arg.lineno - 1
    # `arg.col_offset` points at the opening quote of the string literal.
    # Adding 1 skips the quote; works for normal `'...'` / `"..."`.
    # Triple-quoted or implicit-concat literals can produce slightly
    # off offsets — we accept that as a cosmetic edge case.
    quote_skip = 1
    seg_offset = 0
    for i, seg in enumerate(chain):
        if i == problem.segment_index:
            break
        seg_offset += len(seg) + len("__")
    col_start = arg.col_offset + quote_skip + leading + seg_offset
    col_end = col_start + len(problem.bad_segment)
    return _make_orm_diagnostic(
        line0, col_start, col_end, _format_orm_message(problem), problem
    )


def _format_orm_message(problem: lookup_walker.Problem) -> str:
    if problem.outcome == "unknown_field":
        msg = (
            f"unknown ORM field/relation {problem.bad_segment!r} on "
            f"{problem.on_model}"
        )
        if problem.available:
            hint = ", ".join(problem.available[:8])
            if len(problem.available) > 8:
                hint += ", …"
            msg += f"  (available: {hint})"
        return msg
    if problem.outcome == "unknown_lookup":
        return (
            f"unknown ORM lookup {problem.bad_segment!r} after a leaf field "
            f"on {problem.on_model}"
        )
    return f"invalid ORM lookup chain at {problem.bad_segment!r}"


def _make_orm_diagnostic(
    line: int,
    col_start: int,
    col_end: int,
    message: str,
    problem: lookup_walker.Problem,
) -> Diagnostic:
    return {
        "code": _ORM_LOOKUP_DIAG_CODE,
        "message": message,
        "range": {
            "start": {"line": line, "character": col_start},
            "end": {"line": line, "character": col_end},
        },
        "severity": 2,   # warning — bias toward false negatives
        "source": _ORM_LOOKUP_DIAG_SOURCE,
        "data": {
            "outcome": problem.outcome,
            "on_model": problem.on_model,
            "available": list(problem.available),
        },
    }


def _offset_from_lsp_position(text: str, line: int, character: int) -> int:
    """Convert LSP ``{line, character}`` to a Python ``str`` offset.

    LSP characters are UTF-16 code units; non-BMP code points (emoji)
    count as two. For ASCII Python source — the overwhelming common
    case — this collapses to straight character indexing.
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


def _close_brackets(src: str) -> str:
    """Return the closing tokens needed to balance *src*.

    Best-effort string-aware scan — triple-quoted strings and f-strings
    can confuse the cursor over multiple lines, but the resulting parse
    just fails and completion is suppressed.
    """
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


def _find_marker_call(tree: ast.AST, marker: str) -> ast.Call | None:
    """Return the smallest ``Call`` that has *marker* as a keyword name."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == marker:
                return node
    return None


def _field_completion_items(index, model, partial: str, prefix: str = ""):
    """Yield ``CompletionItem``-shaped dicts for *model*'s queryable names.

    Combines declared fields, FK-id accessors, the ``pk`` alias, and
    reverse-relation accessors. Each item carries ``insertText=name=``
    so accepting a completion drops the cursor right after the ``=``.

    When *prefix* is non-empty (a ``foo__`` chain typed before the cursor),
    the label and insertText include it so the editor replaces the whole
    word — without that, accepting a completion would clobber the chain.
    """
    items: dict[str, dict] = {}
    for name in model.fields:
        items[name] = {
            "label": name,
            "detail": _field_detail(model.fields[name]),
        }
    for name in model.fk_id_accessors:
        items.setdefault(name, {
            "label": name,
            "detail": "FK underlying-column accessor",
        })
    items.setdefault("pk", {"label": "pk", "detail": "primary key alias"})
    for name, source in (index.reverse_relations.get(model.qualname) or {}).items():
        items.setdefault(name, {
            "label": name,
            "detail": f"reverse relation → {source}",
        })

    for name in sorted(items):
        if partial and not name.startswith(partial):
            continue
        item = items[name]
        full = f"{prefix}{name}"
        yield {
            "label": full,
            "kind": 5,  # CompletionItemKind.Field
            "insertText": f"{full}=",
            "detail": item["detail"],
            "data": {"source": "iommi_lsp.orm-kwarg", "model": model.qualname},
        }


def _walk_chain_for_completion(
    index, start: "ModelInfo", chain: list[str]
) -> "ModelInfo | None":
    """Walk *chain* from *start* and return the model to complete on.

    Returns the target model when every segment is a relation (forward
    FK/OneToOne/M2M or reverse accessor) and the chain ends on one. Any
    non-relation segment (concrete field, ``pk``, FK-id accessor, ORM
    lookup name) terminates the walk with ``None`` — there's nothing
    meaningful to complete past a scalar.

    Mirrors the conventions of :mod:`lookup_walker` but is structured for
    completion rather than validation: unknown segments also return
    ``None`` so we don't suppress ty with bogus suggestions.
    """
    current: "ModelInfo | None" = start
    for seg in chain:
        if current is None:
            return None
        if seg == "pk" or seg in current.fk_id_accessors:
            return None
        fi = current.fields.get(seg)
        if fi is None:
            source = index.reverse_source(current.qualname, seg)
            if source is None:
                return None
            current = index.models.get(source)
            continue
        if fi.field_type not in RELATION_FIELD_NAMES:
            return None
        target = fi.target
        if target is None:
            return None
        current = index.models.get(target)
    return current


def _field_detail(fi) -> str:
    if fi.target:
        return f"{fi.field_type} → {fi.target}"
    return fi.field_type


def _enclosing_function(tree: ast.Module, target: ast.AST) -> ast.AST | None:
    target_line = getattr(target, "lineno", None)
    if target_line is None:
        return None
    best: ast.AST | None = None
    best_span = 10**9
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.lineno is None or node.end_lineno is None:
            continue
        if not (node.lineno <= target_line <= node.end_lineno):
            continue
        span = node.end_lineno - node.lineno
        if span < best_span:
            best = node
            best_span = span
    return best
