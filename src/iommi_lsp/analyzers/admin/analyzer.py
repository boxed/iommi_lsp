"""Django admin awareness — model-field validation for ``ModelAdmin`` attrs.

For every ``class FooAdmin(admin.ModelAdmin):`` (or any class that
transitively inherits ``ModelAdmin`` — we use a syntactic test rather
than a real MRO walk, but the project's own bases compose fine in
practice), we figure out the registered model two ways:

* ``@admin.register(Model)`` / ``@register(Model)`` decorator on the class;
* ``admin.site.register(Model, FooAdmin)`` somewhere later in the file
  (module-level scan).

Then any string entry in a recognised admin-list attribute
(``list_display``, ``list_filter``, ``search_fields``, …) is treated as
a Django ORM lookup against that model — same machinery as
``Model.objects.filter(field=…)``. Diagnostics surface as
``django-unknown-admin-field`` warnings; completion offers field names
when the cursor sits inside one of those strings.

Field-lookup admin attrs additionally accept a small set of
non-field tokens — e.g. ``list_display`` can include a method name on
the admin class, ``search_fields`` can prefix with ``=``/``^``/``@``,
``ordering`` can prefix with ``-``. We strip those prefixes before
validating and skip the validation entirely when the string matches a
method on the admin class.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

from ... import log
from ..base import CompletionResult, Diagnostic
from ..django import lookup_walker

if TYPE_CHECKING:
    from ..django.index import DjangoIndex, ModelInfo


_log = log.get("admin.analyzer")


# ---------------------------------------------------------------------------
# Attribute classification.
# ---------------------------------------------------------------------------


# Attrs whose entries are model field paths. Some accept a leading ``-``
# (ordering) or search-field prefixes (``=``, ``^``, ``@``).
_FIELD_LIST_ATTRS: frozenset[str] = frozenset({
    "list_display",
    "list_filter",
    "search_fields",
    "readonly_fields",
    "list_editable",
    "list_select_related",   # accepts True/list; the list entries are field paths
    "ordering",
    "autocomplete_fields",
    "raw_id_fields",
    "filter_horizontal",
    "filter_vertical",
    "fields",
    "exclude",
})

# Attrs whose value is a single field name (string).
_FIELD_SCALAR_ATTRS: frozenset[str] = frozenset({
    "date_hierarchy",
})

# ``fieldsets`` is special: a list of ``(group_name, options_dict)`` where
# ``options_dict['fields']`` is the field list. Handled separately.

# ``prepopulated_fields`` is ``{ 'slug': ('title',) }`` — the dict key is a
# field name and each value list is field names. Handled separately too.

ADMIN_DIAG_CODE = "django-unknown-admin-field"
ADMIN_DIAG_SOURCE = "iommi_lsp"


# ---------------------------------------------------------------------------
# Public analyzer.
# ---------------------------------------------------------------------------


class AdminAnalyzer:
    """Implements the :class:`Analyzer` Protocol for Django admin attrs."""

    name = "admin"

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

    # -- Analyzer protocol ----------------------------------------------------

    async def index(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    async def on_file_changed(self, uri: str) -> None:
        return None

    def is_false_positive(self, uri: str, diagnostic: Diagnostic) -> bool:
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
            _log.exception("admin diagnostic scanner crashed; emitting nothing")
            return []

    def _parse(self, uri: str, source: str) -> ast.Module | None:
        if self._parse_provider is not None:
            return self._parse_provider(uri, source)
        try:
            return ast.parse(source)
        except SyntaxError:
            return None

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
            _log.exception("admin completion scanner crashed; emitting nothing")
            return empty

    # -- internals ------------------------------------------------------------

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
# AST scrape — admin classes & their bound models.
# ---------------------------------------------------------------------------


@dataclass
class _AdminClass:
    cls_node: ast.ClassDef
    model_name: str | None
    methods: set[str] = field(default_factory=set)


def _is_model_admin_base(base: ast.AST) -> bool:
    if isinstance(base, ast.Name) and base.id in {
        "ModelAdmin", "TabularInline", "StackedInline",
    }:
        return True
    if isinstance(base, ast.Attribute) and base.attr in {
        "ModelAdmin", "TabularInline", "StackedInline",
    }:
        return True
    return False


def _model_from_register_decorator(cls_node: ast.ClassDef) -> str | None:
    """``@admin.register(MyModel)`` / ``@register(MyModel)`` — return ``"MyModel"``."""
    for deco in cls_node.decorator_list:
        if isinstance(deco, ast.Call):
            func = deco.func
            if (
                (isinstance(func, ast.Attribute) and func.attr == "register")
                or (isinstance(func, ast.Name) and func.id == "register")
            ):
                if deco.args:
                    arg = deco.args[0]
                    if isinstance(arg, ast.Name):
                        return arg.id
                    if isinstance(arg, ast.Attribute):
                        return arg.attr
    return None


def _model_from_site_register(tree: ast.Module, admin_class_name: str) -> str | None:
    """Scan module statements for ``admin.site.register(Model, FooAdmin)``."""
    for stmt in tree.body:
        if not isinstance(stmt, ast.Expr):
            continue
        call = stmt.value
        if not isinstance(call, ast.Call):
            continue
        # ``admin.site.register(...)`` or ``site.register(...)``.
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "register"):
            continue
        if len(call.args) < 2:
            continue
        model_arg, admin_arg = call.args[0], call.args[1]
        if isinstance(admin_arg, ast.Name) and admin_arg.id == admin_class_name:
            if isinstance(model_arg, ast.Name):
                return model_arg.id
            if isinstance(model_arg, ast.Attribute):
                return model_arg.attr
    return None


def _admin_classes_in(tree: ast.Module) -> list[_AdminClass]:
    out: list[_AdminClass] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(_is_model_admin_base(b) for b in node.bases):
            continue
        model = _model_from_register_decorator(node)
        if model is None:
            model = _model_from_site_register(tree, node.name)
        methods: set[str] = set()
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.add(stmt.name)
        out.append(_AdminClass(cls_node=node, model_name=model, methods=methods))
    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _strip_admin_prefix(attr: str, raw: str) -> str:
    """Remove the leading sigil from *raw* per *attr*-specific rules."""
    if attr == "ordering" and raw.startswith("-"):
        return raw[1:]
    if attr == "search_fields" and raw[:1] in {"=", "^", "@"}:
        return raw[1:]
    return raw


def _validate_string(
    chain_str: str,
    *,
    arg_node: ast.Constant,
    leading: int,
    model: "ModelInfo",
    index: "DjangoIndex",
    admin_methods: set[str],
):
    if not chain_str:
        return None
    # Admin convention: ``list_display`` can include a method name on
    # the admin class itself. Pass silently.
    if chain_str in admin_methods:
        return None
    # Also tolerate ``__str__`` / ``self`` placeholders some teams use.
    if chain_str in {"__str__"}:
        return None
    chain = lookup_walker.split_chain(chain_str)
    result = lookup_walker.walk(index, model.qualname, chain)
    if not isinstance(result, lookup_walker.Problem):
        return None
    if arg_node.lineno is None or arg_node.col_offset is None:
        return None
    line0 = arg_node.lineno - 1
    col_start = arg_node.col_offset + 1 + leading
    seg_offset = 0
    for i, seg in enumerate(chain):
        if i == result.segment_index:
            break
        seg_offset += len(seg) + len("__")
    col_start += seg_offset
    col_end = col_start + len(result.bad_segment)
    msg = (
        f"unknown admin field {result.bad_segment!r} on {result.on_model}"
    )
    if result.available:
        hint = ", ".join(sorted(result.available)[:8])
        if len(result.available) > 8:
            hint += ", …"
        msg += f"  (available: {hint})"
    return {
        "code": ADMIN_DIAG_CODE,
        "message": msg,
        "range": {
            "start": {"line": line0, "character": col_start},
            "end": {"line": line0, "character": col_end},
        },
        "severity": 2,
        "source": ADMIN_DIAG_SOURCE,
        "data": {
            "outcome": result.outcome,
            "on_model": result.on_model,
            "available": list(result.available),
        },
    }


def _iter_field_strings(value: ast.AST, attr: str):
    """Yield ``(string_node, raw_text, leading_offset)`` for every entry of *value*.

    *leading_offset* is the count of source characters from the start of
    the string literal contents (after the opening quote) to the start
    of the field portion — e.g. 1 for the ``-`` in ``ordering=('-foo',)``.
    """
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        for elt in value.elts:
            yield from _iter_field_strings(elt, attr)
        return
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        raw = value.value
        stripped = _strip_admin_prefix(attr, raw)
        leading = len(raw) - len(stripped)
        yield value, stripped, leading


def _scan_diagnostics(tree: ast.Module, index: "DjangoIndex"):
    for cls_info in _admin_classes_in(tree):
        if cls_info.model_name is None:
            continue
        model = index.lookup(cls_info.model_name)
        if model is None:
            continue
        for stmt in cls_info.cls_node.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            tgt = stmt.targets[0]
            if not isinstance(tgt, ast.Name):
                continue
            attr = tgt.id
            if attr in _FIELD_LIST_ATTRS:
                for arg, raw, lead in _iter_field_strings(stmt.value, attr):
                    diag = _validate_string(
                        raw,
                        arg_node=arg,
                        leading=lead,
                        model=model,
                        index=index,
                        admin_methods=cls_info.methods,
                    )
                    if diag is not None:
                        yield diag
            elif attr in _FIELD_SCALAR_ATTRS:
                if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                    diag = _validate_string(
                        stmt.value.value,
                        arg_node=stmt.value,
                        leading=0,
                        model=model,
                        index=index,
                        admin_methods=cls_info.methods,
                    )
                    if diag is not None:
                        yield diag
            elif attr == "fieldsets":
                yield from _validate_fieldsets(
                    stmt.value, model, index, cls_info.methods,
                )
            elif attr == "prepopulated_fields":
                yield from _validate_prepopulated(
                    stmt.value, model, index, cls_info.methods,
                )


def _validate_fieldsets(
    value: ast.AST,
    model: "ModelInfo",
    index: "DjangoIndex",
    admin_methods: set[str],
):
    """``fieldsets = [(label, {'fields': [...]}), ...]``."""
    if not isinstance(value, (ast.List, ast.Tuple)):
        return
    for entry in value.elts:
        if not isinstance(entry, ast.Tuple) or len(entry.elts) < 2:
            continue
        opts = entry.elts[1]
        if not isinstance(opts, ast.Dict):
            continue
        for k, v in zip(opts.keys, opts.values):
            if not (isinstance(k, ast.Constant) and k.value == "fields"):
                continue
            for arg, raw, lead in _iter_field_strings(v, "fields"):
                diag = _validate_string(
                    raw,
                    arg_node=arg,
                    leading=lead,
                    model=model,
                    index=index,
                    admin_methods=admin_methods,
                )
                if diag is not None:
                    yield diag


def _validate_prepopulated(
    value: ast.AST,
    model: "ModelInfo",
    index: "DjangoIndex",
    admin_methods: set[str],
):
    """``prepopulated_fields = {'slug': ('title',)}``."""
    if not isinstance(value, ast.Dict):
        return
    for k, v in zip(value.keys, value.values):
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            diag = _validate_string(
                k.value,
                arg_node=k,
                leading=0,
                model=model,
                index=index,
                admin_methods=admin_methods,
            )
            if diag is not None:
                yield diag
        for arg, raw, lead in _iter_field_strings(v, "fields"):
            diag = _validate_string(
                raw,
                arg_node=arg,
                leading=lead,
                model=model,
                index=index,
                admin_methods=admin_methods,
            )
            if diag is not None:
                yield diag


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------


_MARKER = "__iommi_lsp_admin_marker__"


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

    # AST-patch: replace the open string (from its opening quote up to
    # the cursor) with a known sentinel literal, keeping the rest of the
    # source intact so module-level registrations *below* the cursor
    # (``admin.site.register(Model, FooAdmin)``) still parse.
    suffix_start = _open_string_end(source, ctx, offset)
    head = source[:ctx.start]
    inserted = f'"{_MARKER}"'
    patched = head + inserted + source[suffix_start:]
    try:
        tree = ast.parse(patched)
    except SyntaxError:
        # Fall back to the truncated-and-rebalanced form — handles
        # mid-statement edits where the suffix is genuinely broken.
        closes = _close_brackets(head + inserted)
        try:
            tree = ast.parse(head + inserted + closes)
        except SyntaxError:
            return empty

    info = _find_marker_context(tree)
    if info is None:
        return empty
    attr, cls_node = info

    cls_info = _admin_class_info(cls_node, tree)
    if cls_info is None or cls_info.model_name is None:
        return empty
    model = index.lookup(cls_info.model_name)
    if model is None:
        return empty

    raw_partial = source[ctx.start + 1: offset]
    stripped_partial = _strip_admin_prefix(attr, raw_partial)
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
    candidates = sorted(_field_names_for(model, index))
    for name in candidates:
        if stripped_partial and not name.startswith(stripped_partial):
            continue
        items.append({
            "label": name,
            "kind": 5,   # Field
            "insertText": name,
            "textEdit": {"range": edit_range, "newText": name},
            "detail": f"{model.qualname} ({attr})",
            "data": {"source": "iommi_lsp.admin", "model": model.qualname},
        })
    return CompletionResult(items=items, exclusive=True)


def _field_names_for(model: "ModelInfo", index: "DjangoIndex") -> set[str]:
    out: set[str] = set(model.fields.keys())
    out.update(model.fk_id_accessors)
    out.add("pk")
    out.update(index.reverse_relations.get(model.qualname, {}).keys())
    return out


def _admin_class_info(cls_node: ast.ClassDef, tree: ast.Module) -> _AdminClass | None:
    if not any(_is_model_admin_base(b) for b in cls_node.bases):
        return None
    model = _model_from_register_decorator(cls_node)
    if model is None:
        model = _model_from_site_register(tree, cls_node.name)
    methods: set[str] = set()
    for stmt in cls_node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.add(stmt.name)
    return _AdminClass(cls_node=cls_node, model_name=model, methods=methods)


def _find_marker_context(tree: ast.Module) -> tuple[str, ast.ClassDef] | None:
    """Locate the sentinel and report which admin attr/class it landed in."""
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        if not any(_is_model_admin_base(b) for b in cls.bases):
            continue
        for stmt in cls.body:
            attr = _admin_attr_target(stmt)
            if attr is None:
                continue
            if _contains_marker(stmt):
                return attr, cls
    return None


def _admin_attr_target(stmt: ast.AST) -> str | None:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        t = stmt.targets[0]
        if isinstance(t, ast.Name) and (
            t.id in _FIELD_LIST_ATTRS
            or t.id in _FIELD_SCALAR_ATTRS
            or t.id in {"fieldsets", "prepopulated_fields"}
        ):
            return t.id
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        if stmt.target.id in _FIELD_LIST_ATTRS | _FIELD_SCALAR_ATTRS:
            return stmt.target.id
    return None


def _contains_marker(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and sub.value == _MARKER:
            return True
    return False


# ---------------------------------------------------------------------------
# Generic helpers — mirror of the ones in templates/settings analyzers.
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
    """Return an offset to splice the post-string suffix from.

    Walks forward from *cursor* on the same line looking for an
    un-escaped occurrence of the same quote character (closing the open
    string). If found, returns the position immediately *after* that
    quote. If not found before end-of-line, returns *cursor* — the
    open string genuinely has no terminator yet and the suffix from the
    cursor onward is unrelated source code.
    """
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
