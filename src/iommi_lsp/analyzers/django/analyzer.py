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
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from typing import TYPE_CHECKING

from ... import log
from ..base import Analyzer, Diagnostic
from .index import (
    DjangoIndex,
    ModelInfo,
    _FileScrape,
    assemble_index,
    collect_scrapes,
    update_scrapes,
)
from .magic import FK_LIKE_FIELD_NAMES

if TYPE_CHECKING:
    from ...config import Config


_log = log.get("django.analyzer")


_QUERY_METHODS_RETURNING_INSTANCE = frozenset({
    "get", "first", "last", "earliest", "latest",
    "create", "get_or_create", "update_or_create",
})


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
    ) -> None:
        # Lazy import — config.py pulls in this package via magic.py and we
        # need to break the cycle.
        from ...config import DEFAULT as DEFAULT_CONFIG

        self.workspace_root = workspace_root
        self.django_index: DjangoIndex = django_index or DjangoIndex()
        self.config: "Config" = config or DEFAULT_CONFIG
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
        cached = self._cache.get(uri)
        if cached is not None:
            return cached
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as e:
            _log.debug("could not parse %s: %s", path, e)
            return None
        parsed = _ParsedFile(tree=tree, source=source)
        self._cache[uri] = parsed
        return parsed

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
        if not _is_unresolved_attribute(diagnostic):
            return False
        try:
            return self._evaluate(uri, diagnostic)
        except Exception:
            _log.exception("analyzer crashed; keeping the diagnostic")
            return False


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
