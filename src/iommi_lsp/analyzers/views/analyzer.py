"""Class-based-view awareness.

For every class transitively inheriting Django's generic CBVs
(syntactic check — we look at the base names), we extract:

* ``model = X`` — the bound model, if any;
* ``fields`` — a list of model-field-name strings;
* ``ordering`` — a list of ORM-lookup strings (with optional ``-`` prefix);
* ``slug_field`` — a single model-field-name string.

When the cursor sits inside one of those string positions, we offer the
model's queryable names. Strings that don't validate get a
``django-unknown-view-field`` warning.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

from ... import log
from ..base import CompletionResult, Diagnostic
from ..django import lookup_walker

if TYPE_CHECKING:
    from ..django.index import DjangoIndex, ModelInfo


_log = log.get("views.analyzer")


VIEW_FIELD_DIAG_CODE = "django-unknown-view-field"
DIAG_SOURCE = "iommi_lsp"


_CBV_BASE_NAMES: frozenset[str] = frozenset({
    "View",
    "TemplateView",
    "RedirectView",
    "ListView",
    "DetailView",
    "CreateView",
    "UpdateView",
    "DeleteView",
    "FormView",
    "ModelFormMixin",
    "MultipleObjectMixin",
    "SingleObjectMixin",
    "ArchiveIndexView",
    "YearArchiveView",
    "MonthArchiveView",
    "WeekArchiveView",
    "DayArchiveView",
    "TodayArchiveView",
    "DateDetailView",
    "LoginRequiredMixin",      # often appears as a base, alongside a CBV
    "PermissionRequiredMixin",
    "UserPassesTestMixin",
})


# Attrs whose value is a list/tuple of model-field-name strings.
_FIELD_LIST_ATTRS: frozenset[str] = frozenset({"fields"})

# Attrs whose value is a list/tuple of ORM-lookup strings (``-foo`` allowed).
_ORDERING_LIST_ATTRS: frozenset[str] = frozenset({"ordering"})

# Attrs whose value is a single string field name.
_FIELD_SCALAR_ATTRS: frozenset[str] = frozenset({"slug_field"})


# Class attributes that Django's generic CBV mixins set as defaults
# (``MultipleObjectMixin`` / ``SingleObjectMixin`` / ``TemplateResponseMixin``).
# ty doesn't see these without Django stubs loaded, so accessing
# ``self.paginate_by`` etc. in an override fires an ``unresolved-attribute``
# diagnostic. They're real, stable Django API — suppress when the
# enclosing class transitively inherits a generic CBV.
_CBV_INHERITED_ATTRS: frozenset[str] = frozenset({
    "context_object_name",
    "paginate_by",
    "paginator_class",
    "page_kwarg",
    "slug_url_kwarg",
    "pk_url_kwarg",
    "slug_field",
    "template_name",
    "template_name_suffix",
    "template_name_field",
    "queryset",
    "object",
    "object_list",
    "form_class",
    "success_url",
    "initial",
    "prefix",
    "http_method_names",
    "response_class",
    "content_type",
    "extra_context",
})


@dataclass
class _ViewClass:
    cls_node: ast.ClassDef
    model_name: str | None


def _is_cbv_base(base: ast.AST) -> bool:
    if isinstance(base, ast.Name) and base.id in _CBV_BASE_NAMES:
        return True
    if isinstance(base, ast.Attribute) and base.attr in _CBV_BASE_NAMES:
        return True
    return False


def _model_attr(cls_node: ast.ClassDef) -> str | None:
    for stmt in cls_node.body:
        target_id = None
        value: ast.AST | None = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            if isinstance(stmt.targets[0], ast.Name):
                target_id = stmt.targets[0].id
                value = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            target_id = stmt.target.id
            value = stmt.value
        if target_id == "model" and value is not None:
            if isinstance(value, ast.Name):
                return value.id
            if isinstance(value, ast.Attribute):
                return value.attr
    return None


def _view_class_info(cls_node: ast.ClassDef) -> _ViewClass | None:
    if not any(_is_cbv_base(b) for b in cls_node.bases):
        return None
    return _ViewClass(cls_node=cls_node, model_name=_model_attr(cls_node))


# ---------------------------------------------------------------------------
# Public analyzer.
# ---------------------------------------------------------------------------


class ViewsAnalyzer:
    """Implements the :class:`Analyzer` Protocol for Django CBV awareness."""

    name = "views"

    def __init__(
        self,
        workspace_root: Path,
        text_provider: Callable[[str], str | None] | None = None,
        django_index_provider: "Callable[[], DjangoIndex] | None" = None,
        parse_provider: "Callable[[str, str], ast.Module | None] | None" = None,
    ) -> None:
        self.workspace_root = workspace_root
        self._text_provider = text_provider
        self._django_index_provider = django_index_provider
        self._parse_provider = parse_provider

    async def index(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    async def on_file_changed(self, uri: str) -> None:
        return None

    def is_false_positive(self, uri: str, diagnostic: Diagnostic) -> bool:
        if not _is_unresolved_attribute(diagnostic):
            return False
        path = _uri_to_path(uri)
        if path is None:
            return False
        source = self._source_for(uri, path)
        if source is None:
            return False
        tree = self._parse(uri, source)
        if tree is None:
            return False
        try:
            return _diagnostic_is_cbv_self_attr(tree, diagnostic)
        except Exception:
            _log.exception(
                "CBV self-attr suppression check crashed; keeping the diagnostic"
            )
            return False

    def additional_diagnostics(self, uri: str) -> list[Diagnostic]:
        index = self._index()
        if index is None or not index.models:
            return []
        path = _uri_to_path(uri)
        if path is None:
            return []
        source = self._source_for(uri, path)
        if source is None:
            return []
        tree = self._parse(uri, source)
        if tree is None:
            return []
        try:
            return list(_scan_diagnostics(tree, index))
        except Exception:
            _log.exception("views diagnostic scanner crashed; emitting nothing")
            return []

    def completions(self, uri: str, position: dict) -> CompletionResult:
        empty = CompletionResult()
        index = self._index()
        if index is None or not index.models:
            return empty
        path = _uri_to_path(uri)
        if path is None:
            return empty
        source = self._source_for(uri, path)
        if source is None:
            return empty
        try:
            return _scan_completions(source, position, index)
        except Exception:
            _log.exception("views completion scanner crashed; emitting nothing")
            return empty

    def _parse(self, uri: str, source: str) -> ast.Module | None:
        if self._parse_provider is not None:
            return self._parse_provider(uri, source)
        try:
            return ast.parse(source)
        except SyntaxError:
            return None

    def _index(self) -> "DjangoIndex | None":
        if self._django_index_provider is None:
            return None
        try:
            return self._django_index_provider()
        except Exception:
            return None

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


# ---------------------------------------------------------------------------
# False-positive suppression for ``self.<inherited CBV attr>``
# ---------------------------------------------------------------------------


def _is_unresolved_attribute(diagnostic: Diagnostic) -> bool:
    code = diagnostic.get("code")
    if isinstance(code, str) and code == "unresolved-attribute":
        return True
    if isinstance(code, dict) and code.get("value") == "unresolved-attribute":
        return True
    return False


def _diagnostic_is_cbv_self_attr(tree: ast.Module, diagnostic: Diagnostic) -> bool:
    """Drop the diagnostic when it pins to ``self.<inherited CBV attr>``
    inside a class that transitively inherits a generic CBV.

    AST-only: we look for the smallest ``ast.Attribute`` whose range
    contains the diagnostic's range, check its receiver is ``self``, its
    attr is in :data:`_CBV_INHERITED_ATTRS`, and the enclosing class has
    a recognised CBV base.
    """
    rng = diagnostic.get("range") or {}
    attr = _find_attribute_at(tree, rng)
    if attr is None:
        return False
    if attr.attr not in _CBV_INHERITED_ATTRS:
        return False
    if not (isinstance(attr.value, ast.Name) and attr.value.id == "self"):
        return False
    cls = _enclosing_cbv_class(tree, attr)
    return cls is not None


_ATTR_INDEX_ATTR = "_iommi_lsp_views_attr_index"
_CBV_CLASS_INDEX_ATTR = "_iommi_lsp_views_cbv_classes"


def _attr_index(tree: ast.Module) -> list[ast.Attribute]:
    """All ``ast.Attribute`` nodes in *tree*, computed once per parse."""
    cached = getattr(tree, _ATTR_INDEX_ATTR, None)
    if cached is not None:
        return cached
    attrs = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    try:
        setattr(tree, _ATTR_INDEX_ATTR, attrs)
    except (AttributeError, TypeError):
        pass
    return attrs


def _find_attribute_at(tree: ast.Module, rng: dict) -> ast.Attribute | None:
    start = rng.get("start") or {}
    end = rng.get("end") or {}
    s_line = int(start.get("line", 0)) + 1
    s_col = int(start.get("character", 0))
    e_line = int(end.get("line", s_line - 1)) + 1
    e_col = int(end.get("character", s_col))

    best: ast.Attribute | None = None
    best_size = (10**9, 10**9)
    for node in _attr_index(tree):
        nl = node.lineno
        nc = node.col_offset
        nel = node.end_lineno or nl
        nec = node.end_col_offset or nc
        if (nl, nc) > (s_line, s_col):
            continue
        if (nel, nec) < (e_line, e_col):
            continue
        size = (nel - nl, nec - nc)
        if size < best_size:
            best = node
            best_size = size
    return best


def _cbv_classes(tree: ast.Module) -> list[ast.ClassDef]:
    """Sorted-by-lineno list of CBV-rooted classes, cached per parse."""
    cached = getattr(tree, _CBV_CLASS_INDEX_ATTR, None)
    if cached is not None:
        return cached
    out: list[ast.ClassDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.lineno is None or node.end_lineno is None:
            continue
        if not any(_is_cbv_base(b) for b in node.bases):
            continue
        out.append(node)
    out.sort(key=lambda n: n.lineno)
    try:
        setattr(tree, _CBV_CLASS_INDEX_ATTR, out)
    except (AttributeError, TypeError):
        pass
    return out


def _enclosing_cbv_class(tree: ast.Module, target: ast.AST) -> ast.ClassDef | None:
    target_line = getattr(target, "lineno", None)
    if target_line is None:
        return None
    best: ast.ClassDef | None = None
    best_span = 10**9
    for node in _cbv_classes(tree):
        if node.lineno > target_line:
            break
        if node.end_lineno < target_line:
            continue
        span = node.end_lineno - node.lineno
        if span < best_span:
            best = node
            best_span = span
    return best


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _scan_diagnostics(tree: ast.Module, index: "DjangoIndex"):
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        info = _view_class_info(cls)
        if info is None or info.model_name is None:
            continue
        model = index.lookup(info.model_name)
        if model is None:
            continue
        for stmt in cls.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            t = stmt.targets[0]
            if not isinstance(t, ast.Name):
                continue
            attr = t.id
            if attr in _FIELD_LIST_ATTRS:
                for const, raw, lead in _iter_strings(stmt.value, leading_dash=False):
                    yield from _validate(const, raw, lead, model, index)
            elif attr in _ORDERING_LIST_ATTRS:
                if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                    raw = stmt.value.value
                    lead = 1 if raw.startswith("-") else 0
                    yield from _validate(stmt.value, raw[lead:], lead, model, index)
                else:
                    for const, raw, lead in _iter_strings(stmt.value, leading_dash=True):
                        yield from _validate(const, raw, lead, model, index)
            elif attr in _FIELD_SCALAR_ATTRS:
                if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                    yield from _validate(stmt.value, stmt.value.value, 0, model, index)


def _iter_strings(value: ast.AST, *, leading_dash: bool):
    """Yield (node, stripped, leading) for each string entry of a list/tuple."""
    if not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        return
    for elt in value.elts:
        if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
            continue
        raw = elt.value
        lead = 1 if (leading_dash and raw.startswith("-")) else 0
        yield elt, raw[lead:], lead


def _validate(
    const: ast.Constant,
    chain_str: str,
    leading: int,
    model: "ModelInfo",
    index: "DjangoIndex",
):
    if not chain_str:
        return
    chain = lookup_walker.split_chain(chain_str)
    result = lookup_walker.walk(index, model.qualname, chain)
    if not isinstance(result, lookup_walker.Problem):
        return
    if const.lineno is None or const.col_offset is None:
        return
    line0 = const.lineno - 1
    col_start = const.col_offset + 1 + leading
    seg_offset = 0
    for i, seg in enumerate(chain):
        if i == result.segment_index:
            break
        seg_offset += len(seg) + len("__")
    col_start += seg_offset
    col_end = col_start + len(result.bad_segment)
    msg = (
        f"unknown view field {result.bad_segment!r} on {result.on_model}"
    )
    if result.available:
        hint = ", ".join(sorted(result.available)[:8])
        if len(result.available) > 8:
            hint += ", …"
        msg += f"  (available: {hint})"
    yield {
        "code": VIEW_FIELD_DIAG_CODE,
        "message": msg,
        "range": {
            "start": {"line": line0, "character": col_start},
            "end": {"line": line0, "character": col_end},
        },
        "severity": 2,
        "source": DIAG_SOURCE,
    }


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------


_MARKER = "__iommi_lsp_views_marker__"


def _scan_completions(
    source: str, position: dict, index: "DjangoIndex",
) -> CompletionResult:
    empty = CompletionResult()
    line = int(position.get("line", 0))
    character = int(position.get("character", 0))
    offset = _offset_from_lsp_position(source, line, character)
    if offset > len(source):
        return empty

    ctx = _string_state_at(source, offset)
    if ctx is None:
        return empty

    suffix_start = _open_string_end(source, ctx, offset)
    head = source[:ctx.start]
    inserted = f'"{_MARKER}"'
    patched = head + inserted + source[suffix_start:]
    try:
        tree = ast.parse(patched)
    except SyntaxError:
        closes = _close_brackets(head + inserted)
        try:
            tree = ast.parse(head + inserted + closes)
        except SyntaxError:
            return empty

    location = _find_marker_context(tree)
    if location is None:
        return empty
    attr, cls_node = location
    info = _view_class_info(cls_node)
    if info is None or info.model_name is None:
        return empty
    model = index.lookup(info.model_name)
    if model is None:
        return empty

    raw_partial = source[ctx.start + 1: offset]
    stripped_partial = (
        raw_partial[1:]
        if attr in _ORDERING_LIST_ATTRS and raw_partial.startswith("-")
        else raw_partial
    )
    prefix_offset = len(raw_partial) - len(stripped_partial)
    line_start = source.rfind("\n", 0, offset) + 1
    start_character = _lsp_character_in_line(
        source, line_start, ctx.start + 1 + prefix_offset,
    )
    edit_range = {
        "start": {"line": line, "character": start_character},
        "end": {"line": line, "character": character},
    }

    items: list[dict] = []
    for name in sorted(_field_names_for(model, index)):
        if stripped_partial and not name.startswith(stripped_partial):
            continue
        items.append({
            "label": name,
            "kind": 5,
            "insertText": name,
            "textEdit": {"range": edit_range, "newText": name},
            "detail": f"{model.qualname} ({attr})",
            "data": {"source": "iommi_lsp.views"},
        })
    return CompletionResult(items=items, exclusive=True)


def _field_names_for(model: "ModelInfo", index: "DjangoIndex") -> set[str]:
    out: set[str] = set(model.fields.keys())
    out.update(model.fk_id_accessors)
    out.add("pk")
    out.update(index.reverse_relations.get(model.qualname, {}).keys())
    return out


def _find_marker_context(tree: ast.Module) -> tuple[str, ast.ClassDef] | None:
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        if not any(_is_cbv_base(b) for b in cls.bases):
            continue
        for stmt in cls.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            t = stmt.targets[0]
            if not isinstance(t, ast.Name):
                continue
            if t.id not in _FIELD_LIST_ATTRS | _ORDERING_LIST_ATTRS | _FIELD_SCALAR_ATTRS:
                continue
            if _contains_marker(stmt.value):
                return t.id, cls
    return None


def _contains_marker(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and sub.value == _MARKER:
            return True
    return False


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _uri_to_path(uri: str) -> Path | None:
    if not uri.startswith("file://"):
        return None
    parsed = urlparse(uri)
    return Path(unquote(parsed.path))


def _offset_from_lsp_position(text: str, line: int, character: int) -> int:
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


def _lsp_character_in_line(text: str, line_start: int, target_offset: int) -> int:
    char_units = 0
    i = line_start
    while i < target_offset:
        ch = text[i]
        char_units += 2 if ord(ch) > 0xFFFF else 1
        i += 1
    return char_units


class _StringCtx:
    __slots__ = ("quote", "start")

    def __init__(self, quote: str, start: int) -> None:
        self.quote = quote
        self.start = start


def _open_string_end(source: str, ctx: _StringCtx, cursor: int) -> int:
    n = len(source)
    i = cursor
    while i < n and source[i] != "\n":
        ch = source[i]
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == ctx.quote:
            return i + 1
        i += 1
    return cursor


def _string_state_at(source: str, offset: int) -> _StringCtx | None:
    line_start = source.rfind("\n", 0, offset) + 1
    line = source[line_start:offset]
    in_string: str | None = None
    string_start_in_line = -1
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if in_string is not None:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == in_string:
                in_string = None
                string_start_in_line = -1
            i += 1
            continue
        if ch in '"\'':
            if i + 2 < n and line[i + 1] == ch and line[i + 2] == ch:
                return None
            in_string = ch
            string_start_in_line = i
            i += 1
            continue
        if ch == "#":
            return None
        i += 1
    if in_string is None or string_start_in_line < 0:
        return None
    return _StringCtx(quote=in_string, start=line_start + string_start_in_line)


def _close_brackets(src: str) -> str:
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
