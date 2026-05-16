"""Django forms awareness — ``Form`` / ``ModelForm`` field-name help.

For every ``class FooForm(forms.Form):`` we extract:

* the form's declared fields (top-level class attributes that look like
  field constructor calls — same heuristic as the Django model index
  uses to spot model fields);
* for ``ModelForm`` subclasses, ``Meta.model`` / ``Meta.fields`` /
  ``Meta.exclude``.

From that we offer:

* ``django-unknown-clean-method`` diagnostics on ``clean_<name>`` methods
  whose ``<name>`` doesn't match any declared form field (catches
  ``clean_emial`` typos).
* ``django-unknown-form-field`` diagnostics on strings inside
  ``Meta.fields`` / ``Meta.exclude`` that don't resolve against the
  bound model.
* completion inside those same strings (model field names).
* completion inside ``self.fields['‸']`` / ``self.cleaned_data['‸']``
  against the form's declared/Meta-provided field names.

Indexing is per-file (AST only) — no cross-file or runtime work.
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


_log = log.get("forms.analyzer")


FORM_FIELD_DIAG_CODE = "django-unknown-form-field"
CLEAN_METHOD_DIAG_CODE = "django-unknown-clean-method"
DIAG_SOURCE = "iommi_lsp"


_FORM_BASE_NAMES: frozenset[str] = frozenset({
    "Form", "ModelForm", "BaseForm", "BaseModelForm",
    "ModelMultipleChoiceField",   # not a form base, but harmless to skip
})


# ``ModelForm`` bases — used to decide whether Meta.model/.fields/.exclude
# semantics apply. Anything else is treated as a plain ``Form``.
_MODEL_FORM_BASE_NAMES: frozenset[str] = frozenset({
    "ModelForm", "BaseModelForm",
})


# ``Meta`` dict-style attrs whose keys are model field names. The values
# are widget classes / strings / dicts / etc. — we don't validate them;
# only the keys.
_META_DICT_ATTRS: frozenset[str] = frozenset({
    "widgets", "labels", "help_texts", "error_messages", "field_classes",
})


@dataclass
class _FormClass:
    cls_node: ast.ClassDef
    is_model_form: bool
    declared_fields: set[str]               # class attrs that look like ``= forms.Foo(...)``
    meta_model: str | None                  # ``Meta.model = ModelName``
    meta_fields: list[ast.Constant] | None  # node refs to each entry in Meta.fields
    meta_exclude: list[ast.Constant] | None
    meta_use_all: bool                      # ``Meta.fields = '__all__'``
    methods: set[str]
    # ``Meta.widgets`` / ``labels`` / ``help_texts`` / ``error_messages`` /
    # ``field_classes`` — each maps attr name → list of dict-key Constants.
    meta_dict_keys: dict[str, list[ast.Constant]] = field(default_factory=dict)


class FormsAnalyzer:
    """Implements the :class:`Analyzer` Protocol for Django Form awareness."""

    name = "forms"

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
        return False

    def additional_diagnostics(self, uri: str) -> list[Diagnostic]:
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
            return list(_scan_diagnostics(tree, self._index()))
        except Exception:
            _log.exception("forms diagnostic scanner crashed; emitting nothing")
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
        path = _uri_to_path(uri)
        if path is None:
            return empty
        source = self._source_for(uri, path)
        if source is None:
            return empty
        try:
            return _scan_completions(source, position, self._index())
        except Exception:
            _log.exception("forms completion scanner crashed; emitting nothing")
            return empty

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
# AST scrape — form classes.
# ---------------------------------------------------------------------------


def _is_form_base(base: ast.AST) -> bool:
    if isinstance(base, ast.Name) and base.id in _FORM_BASE_NAMES:
        return True
    if isinstance(base, ast.Attribute) and base.attr in _FORM_BASE_NAMES:
        return True
    return False


def _is_model_form_base(base: ast.AST) -> bool:
    if isinstance(base, ast.Name) and base.id in _MODEL_FORM_BASE_NAMES:
        return True
    if isinstance(base, ast.Attribute) and base.attr in _MODEL_FORM_BASE_NAMES:
        return True
    return False


def _looks_like_field_call(value: ast.AST) -> bool:
    """``forms.CharField(...)``, ``CharField(...)`` — anything with ``Field`` in the
    callee name. Conservative match to avoid sweeping in unrelated calls."""
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    name: str | None = None
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute):
        name = func.attr
    if name is None:
        return False
    return name.endswith("Field") or name.endswith("ChoiceField")


def _extract_meta(cls_node: ast.ClassDef):
    """Return the inner ``Meta`` class def, or None."""
    for stmt in cls_node.body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == "Meta":
            return stmt
    return None


def _string_constants_in(node: ast.AST | None) -> list[ast.Constant]:
    if node is None:
        return []
    out: list[ast.Constant] = []
    if isinstance(node, (ast.List, ast.Tuple)):
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt)
    return out


def _dict_string_keys(node: ast.AST | None) -> list[ast.Constant]:
    """Return string-literal key nodes from a dict literal."""
    if not isinstance(node, ast.Dict):
        return []
    out: list[ast.Constant] = []
    for key in node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            out.append(key)
    return out


def _form_class_info(cls_node: ast.ClassDef) -> _FormClass | None:
    if not any(_is_form_base(b) for b in cls_node.bases):
        return None
    is_model_form = any(_is_model_form_base(b) for b in cls_node.bases)
    declared_fields: set[str] = set()
    methods: set[str] = set()
    for stmt in cls_node.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            t = stmt.targets[0]
            if isinstance(t, ast.Name) and _looks_like_field_call(stmt.value):
                declared_fields.add(t.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.value is not None and _looks_like_field_call(stmt.value):
                declared_fields.add(stmt.target.id)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.add(stmt.name)

    meta = _extract_meta(cls_node)
    meta_model: str | None = None
    meta_fields: list[ast.Constant] | None = None
    meta_exclude: list[ast.Constant] | None = None
    meta_use_all = False
    meta_dict_keys: dict[str, list[ast.Constant]] = {}
    if meta is not None:
        for stmt in meta.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                t = stmt.targets[0]
                if isinstance(t, ast.Name):
                    if t.id == "model":
                        meta_model = _resolve_model_arg(stmt.value)
                    elif t.id == "fields":
                        if (
                            isinstance(stmt.value, ast.Constant)
                            and stmt.value.value == "__all__"
                        ):
                            meta_use_all = True
                            meta_fields = []
                        else:
                            meta_fields = _string_constants_in(stmt.value)
                    elif t.id == "exclude":
                        meta_exclude = _string_constants_in(stmt.value)
                    elif t.id in _META_DICT_ATTRS:
                        meta_dict_keys[t.id] = _dict_string_keys(stmt.value)
    return _FormClass(
        cls_node=cls_node,
        is_model_form=is_model_form,
        declared_fields=declared_fields,
        meta_model=meta_model,
        meta_fields=meta_fields,
        meta_exclude=meta_exclude,
        meta_use_all=meta_use_all,
        methods=methods,
        meta_dict_keys=meta_dict_keys,
    )


def _resolve_model_arg(value: ast.AST) -> str | None:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value.rsplit(".", 1)[-1]
    return None


def _effective_field_names(
    info: _FormClass, model: "ModelInfo | None",
) -> set[str]:
    """Compute the form's effective field names — what ``self.fields`` exposes.

    For a ``ModelForm`` this is ``Meta.fields`` (minus ``Meta.exclude``)
    plus the form-class declared overrides. For a plain ``Form``, just
    the declared fields. When ``Meta.fields == '__all__'`` we pull every
    model field (best effort — we don't filter editability since we
    can't tell at AST-time).
    """
    names: set[str] = set(info.declared_fields)
    if info.is_model_form and model is not None:
        if info.meta_use_all:
            names.update(model.fields.keys())
        elif info.meta_fields is not None:
            for c in info.meta_fields:
                if isinstance(c.value, str):
                    names.add(c.value)
        if info.meta_exclude is not None:
            for c in info.meta_exclude:
                if isinstance(c.value, str):
                    names.discard(c.value)
    return names


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _scan_diagnostics(tree: ast.Module, index: "DjangoIndex | None"):
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        info = _form_class_info(cls)
        if info is None:
            continue
        model = (
            index.lookup(info.meta_model) if index and info.meta_model else None
        )
        effective = _effective_field_names(info, model)

        # 1. Meta.fields / Meta.exclude entries against the model.
        if info.is_model_form and model is not None:
            for nodes in (info.meta_fields or [], info.meta_exclude or []):
                for const in nodes:
                    if not isinstance(const.value, str):
                        continue
                    name = const.value
                    if name == "__all__":
                        continue
                    chain = lookup_walker.split_chain(name)
                    result = lookup_walker.walk(index, model.qualname, chain)
                    if isinstance(result, lookup_walker.Problem):
                        diag = _string_diag(
                            const,
                            FORM_FIELD_DIAG_CODE,
                            f"unknown form field {result.bad_segment!r} "
                            f"on {result.on_model}",
                            result,
                        )
                        if diag is not None:
                            yield diag

            # 1b. Meta.widgets / labels / help_texts / error_messages /
            # field_classes — dict keys are model field names.
            for attr_name, keys in info.meta_dict_keys.items():
                for const in keys:
                    if not isinstance(const.value, str):
                        continue
                    chain = lookup_walker.split_chain(const.value)
                    result = lookup_walker.walk(index, model.qualname, chain)
                    if isinstance(result, lookup_walker.Problem):
                        diag = _string_diag(
                            const,
                            FORM_FIELD_DIAG_CODE,
                            f"unknown form field {result.bad_segment!r} "
                            f"in Meta.{attr_name} on {result.on_model}",
                            result,
                        )
                        if diag is not None:
                            yield diag

        # 2. clean_<field> methods. A method named ``clean_<x>`` is
        # expected to validate the form field ``x``. If the form has no
        # field called ``x``, that's almost always a typo.
        for stmt in info.cls_node.body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not stmt.name.startswith("clean_"):
                continue
            if stmt.name == "clean":
                continue
            field_name = stmt.name[len("clean_"):]
            if not field_name:
                continue
            if field_name in effective:
                continue
            # Empty effective set on a ModelForm with no Meta means we
            # can't know what fields exist — stay silent.
            if not effective:
                continue
            diag = _clean_method_diag(stmt, field_name, effective)
            if diag is not None:
                yield diag


def _string_diag(
    const: ast.Constant,
    code: str,
    message: str,
    problem: lookup_walker.Problem,
) -> Diagnostic | None:
    if const.lineno is None or const.col_offset is None:
        return None
    line0 = const.lineno - 1
    col_start = const.col_offset + 1   # skip opening quote
    chain = lookup_walker.split_chain(str(const.value))
    seg_offset = 0
    for i, seg in enumerate(chain):
        if i == problem.segment_index:
            break
        seg_offset += len(seg) + len("__")
    col_start += seg_offset
    col_end = col_start + len(problem.bad_segment)
    return {
        "code": code,
        "message": message,
        "range": {
            "start": {"line": line0, "character": col_start},
            "end": {"line": line0, "character": col_end},
        },
        "severity": 2,
        "source": DIAG_SOURCE,
    }


def _clean_method_diag(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, field_name: str, available: set[str],
) -> Diagnostic | None:
    # Pin to the method name token. ``def `` is 4 chars, so the name
    # starts at col_offset + 4 in 1-indexed AST coords.
    if fn.lineno is None or fn.col_offset is None:
        return None
    line0 = fn.lineno - 1
    # Find "def " followed by the method name.
    name_col = fn.col_offset + len("def ")
    if isinstance(fn, ast.AsyncFunctionDef):
        name_col = fn.col_offset + len("async def ")
    col_start = name_col + len("clean_")
    col_end = name_col + len(fn.name)
    msg = (
        f"clean_{field_name}() refers to a form field {field_name!r} "
        f"that isn't declared"
    )
    if available:
        hint = ", ".join(sorted(available)[:6])
        if len(available) > 6:
            hint += ", …"
        msg += f"  (declared: {hint})"
    return {
        "code": CLEAN_METHOD_DIAG_CODE,
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


_MARKER = "__iommi_lsp_forms_marker__"


def _parse_with_marker(head: str, suffix: str) -> ast.Module | None:
    """Try patching *head* + marker + *suffix* into a parseable tree.

    The straight-line `"__MARKER__"` patch covers most contexts (list /
    tuple / set / value position / subscript). For an unclosed dict
    literal at a key position — ``widgets = {'em`` — the marker has to
    sit on the *key* side of a ``:`` for the AST to parse as a dict
    (otherwise we get ``{"…MARKER…"}`` which parses as a set or a
    mixed-content SyntaxError when there are sibling ``k: v`` pairs).
    """
    plain = f'"{_MARKER}"'
    patched = head + plain + suffix
    try:
        return ast.parse(patched)
    except SyntaxError:
        pass

    closes = _close_brackets(head + plain)
    try:
        return ast.parse(head + plain + closes)
    except SyntaxError:
        pass

    # Dict-key form — preserves a ``{"k": v, "MARKER": None`` parse.
    dict_form = f'"{_MARKER}": None'
    closes_dict = _close_brackets(head + dict_form)
    try:
        return ast.parse(head + dict_form + closes_dict)
    except SyntaxError:
        return None


def _scan_completions(
    source: str, position: dict, index: "DjangoIndex | None",
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
    tree = _parse_with_marker(head, source[suffix_start:])
    if tree is None:
        return empty

    location = _find_marker_location(tree)
    if location is None:
        return empty
    kind, cls_node = location
    info = _form_class_info(cls_node)
    if info is None:
        return empty

    partial = source[ctx.start + 1: offset]
    line_start = source.rfind("\n", 0, offset) + 1
    start_character = _lsp_character_in_line(source, line_start, ctx.start + 1)
    edit_range = {
        "start": {"line": line, "character": start_character},
        "end": {"line": line, "character": character},
    }

    if kind == "meta_fields":
        # Inside Meta.fields / Meta.exclude — complete model field names.
        if not info.is_model_form or not info.meta_model:
            return empty
        if index is None:
            return empty
        model = index.lookup(info.meta_model)
        if model is None:
            return empty
        candidates = sorted(set(model.fields.keys()) | {"__all__"})
        items: list[dict] = []
        for name in candidates:
            if partial and not name.startswith(partial):
                continue
            items.append({
                "label": name,
                "kind": 5,
                "insertText": name,
                "textEdit": {"range": edit_range, "newText": name},
                "detail": f"{model.qualname}",
                "data": {"source": "iommi_lsp.forms-meta-field"},
            })
        return CompletionResult(items=items, exclusive=True)

    if kind.startswith("meta_dict:"):
        # Inside ``Meta.widgets`` / ``Meta.labels`` / ``Meta.help_texts`` /
        # ``Meta.error_messages`` / ``Meta.field_classes`` dict keys —
        # complete model field names.
        attr_name = kind.split(":", 1)[1]
        if not info.is_model_form or not info.meta_model:
            return empty
        if index is None:
            return empty
        model = index.lookup(info.meta_model)
        if model is None:
            return empty
        items = []
        for name in sorted(model.fields.keys()):
            if partial and not name.startswith(partial):
                continue
            items.append({
                "label": name,
                "kind": 5,
                "insertText": name,
                "textEdit": {"range": edit_range, "newText": name},
                "detail": f"{model.qualname} (Meta.{attr_name})",
                "data": {"source": "iommi_lsp.forms-meta-dict"},
            })
        return CompletionResult(items=items, exclusive=True)

    if kind == "self_fields":
        # ``self.fields['‸']`` / ``self.cleaned_data['‸']`` — complete
        # the effective field set.
        model = (
            index.lookup(info.meta_model) if index and info.meta_model else None
        )
        effective = _effective_field_names(info, model)
        if not effective:
            return empty
        items = []
        for name in sorted(effective):
            if partial and not name.startswith(partial):
                continue
            items.append({
                "label": name,
                "kind": 5,
                "insertText": name,
                "textEdit": {"range": edit_range, "newText": name},
                "detail": "form field",
                "data": {"source": "iommi_lsp.forms-self-field"},
            })
        return CompletionResult(items=items, exclusive=True)

    return empty


def _find_marker_location(
    tree: ast.Module,
) -> tuple[str, ast.ClassDef] | None:
    """Walk the patched tree and figure out where the sentinel landed."""
    # Pre-locate the marker constant.
    marker_node: ast.Constant | None = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and n.value == _MARKER:
            marker_node = n
            break
    if marker_node is None:
        return None
    marker_id = id(marker_node)

    # Case A: marker is inside ``self.fields['‸']`` / ``self.cleaned_data['‸']``.
    for sub in ast.walk(tree):
        if not isinstance(sub, ast.Subscript):
            continue
        if not _is_self_fields_or_cleaned(sub.value):
            continue
        if _subtree_contains(sub.slice, marker_id):
            cls = _enclosing_class(tree, sub)
            if cls is not None:
                return "self_fields", cls

    # Case B: marker is inside an inner Meta class's ``fields`` /
    # ``exclude`` assignment value.
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        meta = _extract_meta(cls)
        if meta is None:
            continue
        for stmt in meta.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for t in stmt.targets:
                if not isinstance(t, ast.Name):
                    continue
                if t.id in {"fields", "exclude"}:
                    if _subtree_contains(stmt.value, marker_id):
                        return "meta_fields", cls
                elif t.id in _META_DICT_ATTRS:
                    # Only fire for the *key* side of a dict literal —
                    # values are widget classes / strings / dicts that the
                    # user clearly isn't typing a field name into.
                    if _marker_is_dict_key(stmt.value, marker_id):
                        return f"meta_dict:{t.id}", cls
    return None


def _marker_is_dict_key(node: ast.AST, marker_id: int) -> bool:
    """Is the marker constant a top-level key of *node* (an ast.Dict)?

    Also accepts the set-shaped fallback (``{"MARKER"}``) for the
    unclosed-dict case where the patched source becomes a set literal.
    """
    if isinstance(node, ast.Set):
        for elt in node.elts:
            if id(elt) == marker_id:
                return True
    if not isinstance(node, ast.Dict):
        return False
    for k in node.keys:
        if k is not None and id(k) == marker_id:
            return True
    return False


def _is_self_fields_or_cleaned(node: ast.AST) -> bool:
    if not isinstance(node, ast.Attribute):
        return False
    if node.attr not in {"fields", "cleaned_data"}:
        return False
    return isinstance(node.value, ast.Name) and node.value.id == "self"


def _subtree_contains(root: ast.AST, target_id: int) -> bool:
    for n in ast.walk(root):
        if id(n) == target_id:
            return True
    return False


def _enclosing_class(tree: ast.Module, target: ast.AST) -> ast.ClassDef | None:
    tgt_line = getattr(target, "lineno", None)
    if tgt_line is None:
        return None
    best: ast.ClassDef | None = None
    best_span = 10**9
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.lineno is None or node.end_lineno is None:
            continue
        if not (node.lineno <= tgt_line <= node.end_lineno):
            continue
        span = node.end_lineno - node.lineno
        if span < best_span:
            best = node
            best_span = span
    return best


# ---------------------------------------------------------------------------
# Generic helpers.
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
