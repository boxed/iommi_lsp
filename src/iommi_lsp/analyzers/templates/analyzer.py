"""Template-name completion inside string literals + Django-template scanning.

Two surfaces:

* Python string literals — the original form. We walk the workspace at
  init time and collect every file under a ``templates/`` directory
  (Django's app-templates convention). When the cursor sits inside a
  single-line string literal whose pre-cursor content already contains a
  ``/``, we offer every known template that starts with that prefix.
  ``static('foo.css')`` calls additionally complete static files without
  the ``/`` heuristic.

* Django template tags — ``{% url 'name' %}``, ``{% include 'tpl' %}``,
  ``{% extends 'tpl' %}``, ``{% block name %}``, ``{% load tag_lib %}``,
  ``{% static 'foo.css' %}``. Activated when the active document URI has
  a template-ish file extension. Each tag has its own completion logic
  (URL names from the URL index, template names from our template index,
  block names from the parent template via ``extends``, templatetags
  packages from the workspace's ``templatetags/`` modules, static files
  from the static index). ``{% url 'unknown' %}`` also produces
  ``django-unknown-url-name`` diagnostics.

Discovery is one-shot at ``index`` time. New template files / static
files / templatetags packages created after the LSP starts are not
visible until restart (deliberate — keeps the indexer dead simple; the
user mentions this is fine for now).
"""

from __future__ import annotations

import ast
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

from ... import log
from ..base import CompletionResult, Diagnostic

if TYPE_CHECKING:
    from ..urls.analyzer import UrlIndex


_log = log.get("templates.analyzer")


_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "build", "dist", "site-packages",
})


_TEMPLATE_EXTENSIONS = frozenset({
    ".html", ".htm", ".txt", ".xml", ".jinja", ".jinja2", ".j2",
})


# Diagnostic emitted for ``{% url 'unknown' %}``. Mirrors the Python-side
# code from the URL analyzer so editors can suppress them uniformly.
URL_DIAG_CODE = "django-unknown-url-name"
# Diagnostic emitted for ``{% mytag %}`` where ``mytag`` is registered in
# a known templatetags library that the template hasn't ``{% load %}``ed.
UNLOADED_TAG_DIAG_CODE = "django-unloaded-template-tag"
# Same, for ``{{ x|myfilter }}`` uses.
UNLOADED_FILTER_DIAG_CODE = "django-unloaded-template-filter"
DIAG_SOURCE = "iommi_lsp"


# Filters in ``django.template.defaultfilters`` — auto-loaded into every
# template, no ``{% load %}`` required. Sourced from Django's
# ``defaultfilters.py``; bumped manually as Django adds/removes filters.
BUILTIN_FILTERS: frozenset[str] = frozenset({
    "add", "addslashes", "capfirst", "center", "cut", "date", "default",
    "default_if_none", "dictsort", "dictsortreversed", "divisibleby",
    "escape", "escapejs", "escapeseq", "filesizeformat", "first",
    "floatformat", "force_escape", "get_digit", "iriencode", "join",
    "json_script", "last", "length", "length_is", "linebreaks",
    "linebreaksbr", "linenumbers", "ljust", "lower", "make_list",
    "phone2numeric", "pluralize", "pprint", "random", "rjust", "safe",
    "safeseq", "slice", "slugify", "stringformat", "striptags", "time",
    "timesince", "timeuntil", "title", "truncatechars",
    "truncatechars_html", "truncatewords", "truncatewords_html",
    "unordered_list", "upper", "urlencode", "urlize", "urlizetrunc",
    "wordcount", "wordwrap", "yesno",
})


# Tags in ``django.template.defaulttags`` + the loader tags — auto-loaded
# into every template, no ``{% load %}`` required. Includes the
# intermediate tokens (``elif``, ``empty``, …) that Django's parser
# accepts inside their enclosing block tags. Closing tags (``endif``,
# ``endblock``, …) aren't listed — anything starting with ``end`` is
# skipped wholesale by the unloaded-tag scanner since a custom block
# tag's closer (``endmytag``) is implied by its opener.
BUILTIN_TAGS: frozenset[str] = frozenset({
    "autoescape", "block", "comment", "csrf_token", "cycle", "debug",
    "extends", "filter", "firstof", "for", "if", "ifchanged", "include",
    "load", "lorem", "now", "querystring", "regroup", "resetcycle",
    "spaceless", "templatetag", "url", "verbatim", "widthratio", "with",
    # Intermediate tokens of the block tags above (and of i18n's
    # blocktrans, whose ``plural`` arm appears without its library name).
    "elif", "else", "empty", "plural",
})


# Tag libraries that ship with Django itself. They aren't discoverable by
# walking the workspace (they live in site-packages) but the fix for an
# unloaded use is the same ``{% load %}`` — and ``{% static %}`` without
# ``{% load static %}`` is *the* most common instance of this mistake.
DJANGO_LIBRARY_TAGS: dict[str, frozenset[str]] = {
    "static": frozenset({"static", "get_static_prefix", "get_media_prefix"}),
    "i18n": frozenset({
        "trans", "translate", "blocktrans", "blocktranslate", "language",
        "get_available_languages", "get_language_info",
        "get_language_info_list", "get_current_language",
        "get_current_language_bidi",
    }),
    "l10n": frozenset({"localize"}),
    "tz": frozenset({"localtime", "timezone", "get_current_timezone"}),
    "cache": frozenset({"cache"}),
}


# Filter libraries that ship with Django itself — same rationale as
# :data:`DJANGO_LIBRARY_TAGS`. ``{{ n|intcomma }}`` without
# ``{% load humanize %}`` is the classic instance.
DJANGO_LIBRARY_FILTERS: dict[str, frozenset[str]] = {
    "humanize": frozenset({
        "apnumber", "intcomma", "intword", "naturalday", "naturaltime",
        "ordinal",
    }),
    "i18n": frozenset({
        "language_name", "language_name_local", "language_name_translated",
        "language_bidi",
    }),
    "l10n": frozenset({"localize", "unlocalize"}),
    "tz": frozenset({"localtime", "utc", "timezone"}),
}


class TemplateAnalyzer:
    """Implements the :class:`Analyzer` Protocol for template-name completion."""

    name = "templates"

    def __init__(
        self,
        workspace_root: Path,
        text_provider: Callable[[str], str | None] | None = None,
        url_index_provider: "Callable[[], UrlIndex] | None" = None,
    ) -> None:
        self.workspace_root = workspace_root
        self._text_provider = text_provider
        self._url_index_provider = url_index_provider
        self._templates: list[str] = []
        self._statics: list[str] = []
        self._template_paths: dict[str, Path] = {}
        self._templatetags: list[str] = []
        # Library name → set of filter names registered in that library.
        # Populated at index time by parsing each ``templatetags/*.py`` for
        # ``@register.filter`` decorations and ``register.filter('name', fn)``
        # direct calls. Built-in filters from ``django.template.defaultfilters``
        # are not in this dict — they live in :data:`BUILTIN_FILTERS` and are
        # always offered regardless of ``{% load %}`` state.
        self._templatetag_filters: dict[str, set[str]] = {}
        # Library name → set of tag names registered in that library
        # (``@register.tag`` / ``simple_tag`` / ``inclusion_tag`` /
        # ``simple_block_tag``). Drives the unloaded-tag diagnostic and
        # its ``{% load %}`` quick fix.
        self._templatetag_tags: dict[str, set[str]] = {}

    @property
    def templates(self) -> list[str]:
        return list(self._templates)

    @property
    def statics(self) -> list[str]:
        return list(self._statics)

    @property
    def templatetags(self) -> list[str]:
        return list(self._templatetags)

    @property
    def templatetag_filters(self) -> dict[str, set[str]]:
        return {k: set(v) for k, v in self._templatetag_filters.items()}

    @property
    def templatetag_tags(self) -> dict[str, set[str]]:
        return {k: set(v) for k, v in self._templatetag_tags.items()}

    # -- Analyzer protocol ----------------------------------------------------

    async def index(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self._template_paths = discover_templates_with_paths(workspace_root)
        self._templates = sorted(self._template_paths)
        self._statics = sorted(discover_statics(workspace_root))
        self._templatetag_filters, self._templatetag_tags = (
            discover_templatetag_registrations(workspace_root)
        )
        # Templatetags library names are all packages we discovered, even
        # the ones that registered nothing — the user might be mid-edit.
        self._templatetags = sorted(
            set(discover_templatetags(workspace_root))
            | set(self._templatetag_filters)
            | set(self._templatetag_tags)
        )
        filter_total = sum(len(v) for v in self._templatetag_filters.values())
        tag_total = sum(len(v) for v in self._templatetag_tags.values())
        _log.info(
            "indexed %d templates, %d static files, %d templatetags "
            "(%d custom filters, %d custom tags) under %s",
            len(self._templates), len(self._statics),
            len(self._templatetags), filter_total, tag_total, workspace_root,
        )

    async def on_file_changed(self, uri: str) -> None:
        # One-shot discovery — new templates aren't picked up until the
        # LSP restarts. Documented behaviour; revisit when users hit it.
        return None

    def is_false_positive(self, uri: str, diagnostic: Diagnostic) -> bool:
        return False

    def additional_diagnostics(self, uri: str) -> list[Diagnostic]:
        path = _uri_to_path(uri)
        if path is None or not _is_template_file(path):
            return []
        source = self._source_for(uri, path)
        if source is None:
            return []
        out: list[Diagnostic] = []
        url_index = self._url_index()
        if url_index is not None and url_index.entries:
            try:
                out.extend(_template_url_diagnostics(source, url_index))
            except Exception:
                _log.exception("template URL diagnostic scanner crashed; emitting nothing")
        try:
            out.extend(self._unloaded_diagnostics(source))
        except Exception:
            _log.exception("unloaded-tag/filter diagnostic scanner crashed; emitting nothing")
        return out

    def _unloaded_tag_diagnostics(self, source: str) -> list[Diagnostic]:
        """Diagnostics for tags whose library exists but isn't loaded.

        Deliberately conservative: a tag is only flagged when we *know*
        a library that provides it — a workspace ``templatetags/``
        module or one of Django's own (:data:`DJANGO_LIBRARY_TAGS`).
        Tags we can't attribute to any library stay silent; they may
        come from a third-party package in site-packages (which we
        never index), or from a library injected via the ``builtins``
        option of the ``TEMPLATES`` setting.
        """
        providers_by_tag: dict[str, list[str]] = {}
        for lib, tag_names in self._templatetag_tags.items():
            for t in tag_names:
                providers_by_tag.setdefault(t, []).append(lib)
        for lib, tag_names in DJANGO_LIBRARY_TAGS.items():
            for t in tag_names:
                providers_by_tag.setdefault(t, []).append(lib)
        if not providers_by_tag:
            return []

        loaded = _loaded_libraries(source)
        excluded = _excluded_spans(source)
        out: list[Diagnostic] = []
        for m in _ANY_TAG_RE.finditer(source):
            name = m.group(1)
            if name in BUILTIN_TAGS or name.startswith("end"):
                continue
            if any(start <= m.start() < end for start, end in excluded):
                continue
            providers = providers_by_tag.get(name)
            if providers is None:
                continue
            if any(lib in loaded for lib in providers):
                continue
            libraries = sorted(set(providers))
            loads = " or ".join(f"{{% load {lib} %}}" for lib in libraries)
            out.append({
                "code": UNLOADED_TAG_DIAG_CODE,
                "message": f"{name!r} requires {loads}",
                "range": _diag_range(source, m.start(1), m.end(1)),
                "severity": 2,
                "source": DIAG_SOURCE,
                "data": {"tag": name, "libraries": libraries},
            })
        return out

    def _unloaded_filter_diagnostics(self, source: str) -> list[Diagnostic]:
        """Diagnostics for filters whose library exists but isn't loaded.

        Same conservative contract as :meth:`_unloaded_tag_diagnostics`:
        only filters we can attribute to a workspace ``templatetags/``
        module or a Django-bundled library
        (:data:`DJANGO_LIBRARY_FILTERS`) are flagged.
        """
        providers_by_filter: dict[str, list[str]] = {}
        for lib, names in self._templatetag_filters.items():
            for f in names:
                providers_by_filter.setdefault(f, []).append(lib)
        for lib, names in DJANGO_LIBRARY_FILTERS.items():
            for f in names:
                providers_by_filter.setdefault(f, []).append(lib)
        if not providers_by_filter:
            return []

        loaded = _loaded_libraries(source)
        excluded = _excluded_spans(source)
        out: list[Diagnostic] = []
        for name, start_offset in _filter_uses(source):
            if name in BUILTIN_FILTERS:
                continue
            if any(start <= start_offset < end for start, end in excluded):
                continue
            providers = providers_by_filter.get(name)
            if providers is None:
                continue
            if any(lib in loaded for lib in providers):
                continue
            libraries = sorted(set(providers))
            loads = " or ".join(f"{{% load {lib} %}}" for lib in libraries)
            out.append({
                "code": UNLOADED_FILTER_DIAG_CODE,
                "message": f"filter {name!r} requires {loads}",
                "range": _diag_range(source, start_offset, start_offset + len(name)),
                "severity": 2,
                "source": DIAG_SOURCE,
                "data": {"filter": name, "libraries": libraries},
            })
        return out

    def _unloaded_diagnostics(self, source: str) -> list[Diagnostic]:
        return (
            self._unloaded_tag_diagnostics(source)
            + self._unloaded_filter_diagnostics(source)
        )

    def code_actions(self, uri: str, range_: dict, context: dict) -> list[dict]:
        """Quick fixes for unloaded-tag/-filter diagnostics overlapping *range_*.

        One action per candidate library: append the library to the
        template's existing ``{% load %}`` tag when there is one, else
        insert a fresh ``{% load lib %}`` line (after ``{% extends %}``
        when present — extends must stay the first tag).
        """
        path = _uri_to_path(uri)
        if path is None or not _is_template_file(path):
            return []
        source = self._source_for(uri, path)
        if source is None:
            return []
        try:
            diagnostics = self._unloaded_diagnostics(source)
        except Exception:
            _log.exception("unloaded-tag code-action scanner crashed; offering nothing")
            return []
        actions: list[dict] = []
        seen_libs: set[str] = set()
        for diag in diagnostics:
            if not _ranges_overlap(diag["range"], range_):
                continue
            for lib in diag["data"]["libraries"]:
                if lib in seen_libs:
                    # Two unloaded uses of the same library inside the
                    # requested range — one fix loads them both.
                    continue
                seen_libs.add(lib)
                edit, appended = _load_fix_edit(source, lib)
                title = (
                    f"Add '{lib}' to {{% load %}}" if appended
                    else f"Insert {{% load {lib} %}}"
                )
                actions.append({
                    "title": title,
                    "kind": "quickfix",
                    "diagnostics": [diag],
                    "edit": {"changes": {uri: [edit]}},
                })
        return actions

    def completions(self, uri: str, position: dict) -> CompletionResult:
        empty = CompletionResult()
        path = _uri_to_path(uri)
        if path is None:
            return empty
        source = self._source_for(uri, path)
        if source is None:
            return empty
        try:
            if _is_template_file(path):
                return self._template_completions(uri, source, position)
            return self._python_completions(source, position)
        except Exception:
            _log.exception("template completion scanner crashed; emitting nothing")
            return empty

    # -- internals ------------------------------------------------------------

    def _python_completions(self, source: str, position: dict) -> CompletionResult:
        empty = CompletionResult()
        if not self._templates and not self._statics:
            return empty
        line = int(position.get("line", 0))
        character = int(position.get("character", 0))
        offset = _offset_from_lsp_position(source, line, character)
        if offset > len(source):
            return empty

        ctx = _string_state_at(source, offset)
        if ctx is None:
            return empty

        partial = source[ctx.start + 1: offset]

        line_start = source.rfind("\n", 0, offset) + 1
        start_character = _lsp_character_in_line(source, line_start, ctx.start + 1)
        edit_range = {
            "start": {"line": line, "character": start_character},
            "end": {"line": line, "character": character},
        }

        # ``static('foo.css')`` — staticfiles completion. Triggers
        # without the ``/`` heuristic since the call site itself is
        # unambiguous.
        callee = _enclosing_callee(source, ctx.start)
        if callee == "static" and self._statics:
            items: list[dict] = []
            for name in self._statics:
                if partial and not name.startswith(partial):
                    continue
                items.append({
                    "label": name,
                    "kind": 17,   # File
                    "insertText": name,
                    "textEdit": {"range": edit_range, "newText": name},
                    "detail": "static file",
                    "data": {"source": "iommi_lsp.static-name"},
                })
            return CompletionResult(items=items, exclusive=True)

        if "/" not in partial:
            return empty

        # An explicit replacement range is the only way to keep editors
        # that treat `/` as a word boundary (Helix, Neovim's built-in
        # client) from replacing only the trailing word — without it,
        # accepting ``reviews/reviews__tags.html`` on the partial
        # ``reviews/rev`` produces ``reviews/reviews/reviews__tags.html``.
        items = []
        for name in self._templates:
            if not name.startswith(partial):
                continue
            items.append({
                "label": name,
                "kind": 17,   # File
                "insertText": name,
                "textEdit": {"range": edit_range, "newText": name},
                "detail": "template",
                "data": {"source": "iommi_lsp.template-name"},
            })
        # Exclusive when we have matches — otherwise editors with their
        # own path-style completion (Helix's filesystem suggestions, for
        # one) backfill the popup with workspace files (``models.py``,
        # ``__init__.py``, …) that the user clearly isn't reaching for
        # when they've typed a template path. When we have no matches,
        # stay non-exclusive: strings with slashes aren't always
        # templates (URLs, file paths, regex), so let ty's items through.
        if not items:
            return empty
        return CompletionResult(items=items, exclusive=True)

    def _template_completions(
        self, uri: str, source: str, position: dict,
    ) -> CompletionResult:
        empty = CompletionResult()
        line = int(position.get("line", 0))
        character = int(position.get("character", 0))
        offset = _offset_from_lsp_position(source, line, character)
        if offset > len(source):
            return empty

        line_start = source.rfind("\n", 0, offset) + 1

        # Filter position works in both ``{{ x|‸ }}`` (variable) and
        # ``{% if x|‸ %}`` (tag arg) — try the variable form first since
        # ``_enclosing_template_tag`` would otherwise miss it.
        var_range = _enclosing_template_var(source, offset)
        if var_range is not None:
            body_start, body_end = var_range
            filt = self._filter_completion(
                source, offset, body_start, body_end, line, line_start, character,
            )
            if filt is not None:
                return filt
            return empty

        tag = _enclosing_template_tag(source, offset)
        if tag is None:
            return empty

        # Filters can also appear inside ``{% if x|truncatewords:5 %}`` —
        # check before falling through to tag-name dispatch.
        filt = self._filter_completion(
            source, offset, tag.body_start, tag.body_end,
            line, line_start, character,
        )
        if filt is not None:
            return filt

        if tag.name in {"url", "include", "extends", "static"}:
            ctx = _string_state_at_in_range(
                source, offset, tag.body_start, tag.body_end,
            )
            if ctx is None:
                return empty
            partial = source[ctx.start + 1: offset]
            start_character = _lsp_character_in_line(
                source, line_start, ctx.start + 1,
            )
            edit_range = {
                "start": {"line": line, "character": start_character},
                "end": {"line": line, "character": character},
            }
            if tag.name == "url":
                return self._tag_url_completions(partial, edit_range)
            if tag.name == "static":
                return self._tag_static_completions(partial, edit_range)
            # extends / include — template-name completion, no ``/`` heuristic.
            return self._tag_template_completions(partial, edit_range)

        if tag.name == "block":
            return self._tag_block_completions(uri, source, offset, tag, line, line_start)

        if tag.name == "load":
            return self._tag_load_completions(source, offset, tag, line, line_start)

        return empty

    def _tag_url_completions(
        self, partial: str, edit_range: dict,
    ) -> CompletionResult:
        url_index = self._url_index()
        if url_index is None or not url_index.entries:
            return CompletionResult(items=[], exclusive=True)
        items: list[dict] = []
        for name in sorted(url_index.entries):
            if partial and not name.startswith(partial):
                continue
            items.append({
                "label": name,
                "kind": 21,   # Constant
                "insertText": name,
                "textEdit": {"range": edit_range, "newText": name},
                "detail": "URL name ({% url %})",
                "data": {"source": "iommi_lsp.url-name"},
            })
        return CompletionResult(items=items, exclusive=True)

    def _tag_static_completions(
        self, partial: str, edit_range: dict,
    ) -> CompletionResult:
        if not self._statics:
            return CompletionResult(items=[], exclusive=True)
        items: list[dict] = []
        for name in self._statics:
            if partial and not name.startswith(partial):
                continue
            items.append({
                "label": name,
                "kind": 17,   # File
                "insertText": name,
                "textEdit": {"range": edit_range, "newText": name},
                "detail": "static file",
                "data": {"source": "iommi_lsp.static-name"},
            })
        return CompletionResult(items=items, exclusive=True)

    def _tag_template_completions(
        self, partial: str, edit_range: dict,
    ) -> CompletionResult:
        if not self._templates:
            return CompletionResult(items=[], exclusive=True)
        items: list[dict] = []
        for name in self._templates:
            if partial and not name.startswith(partial):
                continue
            items.append({
                "label": name,
                "kind": 17,   # File
                "insertText": name,
                "textEdit": {"range": edit_range, "newText": name},
                "detail": "template",
                "data": {"source": "iommi_lsp.template-name"},
            })
        return CompletionResult(items=items, exclusive=True)

    def _tag_block_completions(
        self, uri: str, source: str, offset: int, tag,
        line: int, line_start: int,
    ) -> CompletionResult:
        # Cursor must be on the bare-word argument after ``block``.
        word_start, word_end = _word_range_at(source, offset, tag.body_start, tag.body_end)
        if word_start < 0:
            return CompletionResult()
        partial = source[word_start: offset]
        block_names = self._parent_block_names(uri, source)
        if not block_names:
            # Recognised position but no parent — stay exclusive so ty's
            # generic completions don't pollute the popup.
            return CompletionResult(items=[], exclusive=True)
        start_character = _lsp_character_in_line(source, line_start, word_start)
        edit_range = {
            "start": {"line": line, "character": start_character},
            "end": {"line": line, "character": _lsp_character_in_line(
                source, line_start, word_end,
            )},
        }
        items: list[dict] = []
        for name in sorted(block_names):
            if partial and not name.startswith(partial):
                continue
            items.append({
                "label": name,
                "kind": 21,   # Constant
                "insertText": name,
                "textEdit": {"range": edit_range, "newText": name},
                "detail": "block (parent template)",
                "data": {"source": "iommi_lsp.block-name"},
            })
        return CompletionResult(items=items, exclusive=True)

    def _filter_completion(
        self, source: str, offset: int, body_start: int, body_end: int,
        line: int, line_start: int, character: int,
    ) -> CompletionResult | None:
        """Return filter completion if cursor is in filter position, else None.

        "Filter position" = the cursor sits on (or right after) an
        identifier whose immediately-preceding non-identifier character is
        ``|``. We don't allow whitespace around ``|`` since Django's own
        filter parser doesn't either.
        """
        rng = _filter_partial_range(source, offset, body_start, body_end)
        if rng is None:
            return None
        word_start, word_end = rng
        partial = source[word_start: offset]

        loaded = _loaded_libraries(source)
        names: set[str] = set(BUILTIN_FILTERS)
        for lib in loaded:
            names |= self._templatetag_filters.get(lib, set())

        start_character = _lsp_character_in_line(source, line_start, word_start)
        end_character = _lsp_character_in_line(source, line_start, word_end)
        edit_range = {
            "start": {"line": line, "character": start_character},
            "end": {"line": line, "character": end_character},
        }
        items: list[dict] = []
        for name in sorted(names):
            if partial and not name.startswith(partial):
                continue
            detail = "filter" if name in BUILTIN_FILTERS else "filter (custom)"
            items.append({
                "label": name,
                "kind": 3,    # Function
                "insertText": name,
                "textEdit": {"range": edit_range, "newText": name},
                "detail": detail,
                "data": {"source": "iommi_lsp.template-filter"},
            })
        return CompletionResult(items=items, exclusive=True)

    def _tag_load_completions(
        self, source: str, offset: int, tag,
        line: int, line_start: int,
    ) -> CompletionResult:
        if not self._templatetags:
            return CompletionResult(items=[], exclusive=True)
        word_start, word_end = _word_range_at(source, offset, tag.body_start, tag.body_end)
        if word_start < 0:
            return CompletionResult()
        partial = source[word_start: offset]
        start_character = _lsp_character_in_line(source, line_start, word_start)
        edit_range = {
            "start": {"line": line, "character": start_character},
            "end": {"line": line, "character": _lsp_character_in_line(
                source, line_start, word_end,
            )},
        }
        items: list[dict] = []
        for name in self._templatetags:
            if partial and not name.startswith(partial):
                continue
            items.append({
                "label": name,
                "kind": 9,   # Module
                "insertText": name,
                "textEdit": {"range": edit_range, "newText": name},
                "detail": "templatetags library",
                "data": {"source": "iommi_lsp.templatetags"},
            })
        return CompletionResult(items=items, exclusive=True)

    def _parent_block_names(self, uri: str, source: str) -> set[str]:
        parent_name = _extends_target(source)
        if parent_name is None:
            return set()
        parent_path = self._template_paths.get(parent_name)
        if parent_path is None:
            return set()
        parent_uri = parent_path.as_uri()
        # Prefer in-editor buffer (text_provider) for the parent if it's
        # also open and being edited; fall back to disk.
        parent_text: str | None = None
        if self._text_provider is not None:
            parent_text = self._text_provider(parent_uri)
        if parent_text is None:
            try:
                parent_text = parent_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                _log.debug("could not read parent template %s: %s", parent_path, e)
                return set()
        names = _block_names_in(parent_text)
        # Recurse one step into the grandparent so grand-blocks are also
        # offered. Bounded to avoid pathological cycles.
        grandparent_name = _extends_target(parent_text)
        if grandparent_name and grandparent_name != parent_name:
            grand_path = self._template_paths.get(grandparent_name)
            if grand_path is not None:
                try:
                    grand_text = grand_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    grand_text = None
                if grand_text is not None:
                    names = names | _block_names_in(grand_text)
        return names

    def _url_index(self):
        if self._url_index_provider is None:
            return None
        try:
            return self._url_index_provider()
        except Exception:
            _log.exception("url_index_provider raised")
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
# Discovery helpers
# ---------------------------------------------------------------------------


def discover_templates(workspace_root: Path) -> set[str]:
    """Return every template name under any ``templates/`` directory.

    A *template name* is the file path relative to the enclosing
    ``templates/`` directory, in POSIX form. Dotfiles and hidden
    directories are skipped; standard noise dirs (``.venv``, ``build``,
    …) are pruned from the search.
    """
    return set(_discover_relative_files(workspace_root, "templates"))


def discover_templates_with_paths(workspace_root: Path) -> dict[str, Path]:
    """Like :func:`discover_templates` but maps each name → absolute path.

    When several apps register the same template name, last-write-wins —
    the resolution order is filesystem-walk order, which isn't quite
    Django's app-priority order, but it's good enough for "look up the
    parent of an extends call" usage. Real apps rarely shadow templates
    on purpose.
    """
    return _discover_relative_files(workspace_root, "templates")


def discover_statics(workspace_root: Path) -> set[str]:
    """Return every static file under any ``static/`` directory."""
    return set(_discover_relative_files(workspace_root, "static"))


def discover_templatetag_filters(workspace_root: Path) -> dict[str, set[str]]:
    """Walk every ``templatetags/`` package and return ``{library: {filter}}``.

    Thin wrapper around :func:`discover_templatetag_registrations` kept
    for callers that only care about filters.
    """
    filters, _tags = discover_templatetag_registrations(workspace_root)
    return filters


def discover_templatetag_registrations(
    workspace_root: Path,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Walk every ``templatetags/`` package; return ``(filters, tags)`` maps.

    Both are ``{library: {name}}``. For each ``templatetags/<lib>.py`` we
    AST-parse the module and pick out registrations:

    * ``@register.filter`` / ``@register.tag`` / ``@register.simple_tag``
      / ``@register.inclusion_tag(...)`` / ``@register.simple_block_tag``
      (bare or called) — uses the explicit ``name=`` (or, for ``filter``
      and ``tag``, the positional string) when given, else the function
      name. ``inclusion_tag``'s positional argument is the template
      filename, never the name.
    * ``register.filter('name', fn)`` / ``register.tag('name', fn)`` —
      direct call form.

    Libraries that registered nothing are omitted (the
    ``discover_templatetags`` set still includes them so the
    ``{% load %}`` popup stays useful).
    """
    filters: dict[str, set[str]] = {}
    tags: dict[str, set[str]] = {}
    root = workspace_root.resolve()
    pkg_dirs: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        if "templatetags" in dirnames:
            pkg_dirs.append(Path(dirpath) / "templatetags")
    for pkg in pkg_dirs:
        for entry in pkg.iterdir():
            if not entry.is_file():
                continue
            if entry.name.startswith("."):
                continue
            if entry.suffix != ".py":
                continue
            stem = entry.stem
            if stem.startswith("_"):
                continue
            lib_filters, lib_tags = _parse_registrations(entry)
            # If two apps register a library under the same name (rare
            # but Django allows it — last loaded wins), union the sets
            # so we don't lose either side's completions.
            if lib_filters:
                filters.setdefault(stem, set()).update(lib_filters)
            if lib_tags:
                tags.setdefault(stem, set()).update(lib_tags)
    return filters, tags


# ``register.<attr>`` decorations that register a *tag* (vs. a filter).
_TAG_REGISTRATION_ATTRS = frozenset({
    "tag", "simple_tag", "inclusion_tag", "simple_block_tag",
    # Pre-1.9 spelling, still floating around in old codebases.
    "assignment_tag",
})


def _parse_registrations(path: Path) -> tuple[set[str], set[str]]:
    """Return ``(filter_names, tag_names)`` registered in module *path*."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return set(), set()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return set(), set()
    filters: set[str] = set()
    tags: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                name = _filter_name_from_decorator(dec, node.name)
                if name is not None:
                    filters.add(name)
                name = _tag_name_from_decorator(dec, node.name)
                if name is not None:
                    tags.add(name)
            continue
        if isinstance(node, ast.Call):
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in ("filter", "tag"):
                continue
            # ``register.filter('name', fn)`` / ``register.tag('name', fn)``
            # require *both* a string name and a callable arg — without the
            # callable it's actually a decorator factory
            # (``@register.filter('name')``) which is already handled above
            # when walking decorator_list.
            if len(node.args) >= 2:
                a0 = node.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    (filters if func.attr == "filter" else tags).add(a0.value)
    return filters, tags


def _filter_name_from_decorator(dec: ast.AST, func_name: str) -> str | None:
    """Resolve the registered name of a ``@register.filter`` decoration."""
    if isinstance(dec, ast.Attribute) and dec.attr == "filter":
        return func_name
    if isinstance(dec, ast.Call):
        func = dec.func
        if not (isinstance(func, ast.Attribute) and func.attr == "filter"):
            return None
        if dec.args:
            a0 = dec.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                return a0.value
        for kw in dec.keywords:
            if kw.arg == "name" and isinstance(kw.value, ast.Constant) \
                    and isinstance(kw.value.value, str):
                return kw.value.value
        return func_name
    return None


def _tag_name_from_decorator(dec: ast.AST, func_name: str) -> str | None:
    """Resolve the registered name of a tag decoration, or None.

    Handles ``@register.tag`` / ``@register.simple_tag`` /
    ``@register.inclusion_tag('tpl.html')`` / ``@register.simple_block_tag``
    in bare and called forms. An explicit ``name=`` kwarg always wins;
    ``@register.tag('name')`` also accepts the name positionally. The
    positional argument of ``inclusion_tag`` is the template filename and
    is deliberately ignored.
    """
    if isinstance(dec, ast.Attribute) and dec.attr in _TAG_REGISTRATION_ATTRS:
        return func_name
    if isinstance(dec, ast.Call):
        func = dec.func
        if not (isinstance(func, ast.Attribute) and func.attr in _TAG_REGISTRATION_ATTRS):
            return None
        for kw in dec.keywords:
            if kw.arg == "name" and isinstance(kw.value, ast.Constant) \
                    and isinstance(kw.value.value, str):
                return kw.value.value
        if func.attr == "tag" and dec.args:
            a0 = dec.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                return a0.value
        return func_name
    return None


def discover_templatetags(workspace_root: Path) -> set[str]:
    """Return templatetags library names across the workspace.

    Walks for ``templatetags/`` packages (Django convention: any app
    with a ``templatetags/`` subpackage exposes its ``.py`` modules as
    ``{% load name %}`` libraries). Dunders (``__init__``, ``__main__``)
    and dotfiles are skipped.
    """
    out: set[str] = set()
    root = workspace_root.resolve()
    pkg_dirs: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        if "templatetags" in dirnames:
            pkg_dirs.append(Path(dirpath) / "templatetags")
            # Don't prune the dir from further recursion — rare, but a
            # ``templatetags/`` package could itself contain another app
            # path. Cheap to keep walking.
    for pkg in pkg_dirs:
        for entry in pkg.iterdir():
            if not entry.is_file():
                continue
            if entry.name.startswith("."):
                continue
            if entry.suffix != ".py":
                continue
            stem = entry.stem
            if stem.startswith("_"):
                continue
            out.add(stem)
    return out


def _discover_relative_files(workspace_root: Path, target_dirname: str) -> dict[str, Path]:
    """Walk *workspace_root* and return ``{posix_relative_name: absolute_path}``.

    Hits every directory named *target_dirname* (typically ``templates``
    or ``static``) and indexes its contents recursively. Dotfiles and
    hidden subdirs are skipped; standard noise dirs are pruned from the
    walk for speed.
    """
    out: dict[str, Path] = {}
    root = workspace_root.resolve()
    asset_dirs: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        if target_dirname in dirnames:
            asset_dirs.append(Path(dirpath) / target_dirname)
            dirnames.remove(target_dirname)
    for adir in asset_dirs:
        for sub_dir, sub_dirnames, sub_files in os.walk(adir):
            sub_dirnames[:] = [d for d in sub_dirnames if not d.startswith(".")]
            for name in sub_files:
                if name.startswith("."):
                    continue
                full = Path(sub_dir, name)
                rel = full.relative_to(adir).as_posix()
                out[rel] = full
    return out


def _enclosing_callee(source: str, string_start: int) -> str | None:
    """Return the callable identifier wrapping the string at *string_start*.

    Cheap left-scan: walk back over whitespace then over an identifier;
    require an opening ``(`` between the identifier and the string.
    """
    i = string_start - 1
    while i >= 0 and source[i] in " \t\r\n":
        i -= 1
    if i < 0 or source[i] != "(":
        return None
    j = i - 1
    end = j + 1
    while j >= 0 and (source[j].isalnum() or source[j] == "_"):
        j -= 1
    name = source[j + 1: end]
    return name or None


def _is_template_file(path: Path) -> bool:
    return path.suffix.lower() in _TEMPLATE_EXTENSIONS


def _uri_to_path(uri: str) -> Path | None:
    if not uri.startswith("file://"):
        return None
    parsed = urlparse(uri)
    return Path(unquote(parsed.path))


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


def _lsp_character_in_line(text: str, line_start: int, target_offset: int) -> int:
    """Return the UTF-16 character offset of *target_offset* within its line.

    Inverse of :func:`_offset_from_lsp_position` for the character axis
    when both points are known to be on the same line — used to build
    LSP ranges from Python ``str`` offsets.
    """
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


def _string_state_at(source: str, offset: int) -> _StringCtx | None:
    """Return the open single-line string at *offset*, or None.

    Only the cursor's own line is scanned: single-line strings can't
    cross a newline, so any quote that opens on a previous line is
    either irrelevant (already closed) or part of a multiline literal
    we deliberately don't handle. A line-local scan also sidesteps the
    triple-quoted-docstring trap — earlier ``\"\"\"…\"\"\"`` blocks no
    longer poison the state for the rest of the file.

    Triple quotes that open on the cursor's own line are still
    ambiguous (we'd misparse ``\"\"\"foo`` as a single-quoted empty
    string followed by an open quote), so we bail in that narrow case.
    """
    line_start = source.rfind("\n", 0, offset) + 1   # 0 when no \n yet
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
            if (
                i + 2 < n
                and line[i + 1] == ch
                and line[i + 2] == ch
            ):
                return None
            in_string = ch
            string_start_in_line = i
            i += 1
            continue
        if ch == "#":
            return None   # comment — rest of line isn't code
        i += 1
    if in_string is None or string_start_in_line < 0:
        return None
    return _StringCtx(quote=in_string, start=line_start + string_start_in_line)


def _string_state_at_in_range(
    source: str, offset: int, body_start: int, body_end: int,
) -> _StringCtx | None:
    """Return the open string literal at *offset* within ``[body_start, body_end)``.

    Used inside template tag bodies (``{% url 'foo|' %}``). Walks left
    from offset over the tag body looking for an unmatched ``'`` or
    ``"``. Returns the position of the opening quote, or None if the
    cursor isn't inside a string literal.
    """
    if offset < body_start or offset > body_end:
        return None
    in_string: str | None = None
    string_start = -1
    i = body_start
    while i < offset:
        ch = source[i]
        if in_string is not None:
            if ch == "\\" and i + 1 < offset:
                i += 2
                continue
            if ch == in_string:
                in_string = None
                string_start = -1
            i += 1
            continue
        if ch in '"\'':
            in_string = ch
            string_start = i
        i += 1
    if in_string is None or string_start < 0:
        return None
    return _StringCtx(quote=in_string, start=string_start)


def _word_range_at(
    source: str, offset: int, body_start: int, body_end: int,
) -> tuple[int, int]:
    """Return ``(start, end)`` of the bare-word at *offset* within the tag body.

    A bare-word is ``[A-Za-z_][A-Za-z0-9_]*`` (Django tag-arg style — no
    quotes, no dots). When the cursor sits at the very end of one (e.g.
    ``{% block fo|``), end is the offset itself; the right-extension lets
    us replace the partial cleanly.

    Returns ``(-1, -1)`` if the cursor isn't on a bare-word position
    (e.g. inside a quoted string, or sitting on whitespace right after
    the tag name).
    """
    if offset < body_start or offset > body_end:
        return -1, -1
    # Don't fire when cursor is mid-string.
    s = _string_state_at_in_range(source, offset, body_start, body_end)
    if s is not None:
        return -1, -1
    start = offset
    while start > body_start and _is_word_char(source[start - 1]):
        start -= 1
    end = offset
    while end < body_end and _is_word_char(source[end]):
        end += 1
    if start == end:
        # Allow an empty word right after whitespace — caller may still
        # offer the full list. Return zero-length range at offset.
        return offset, offset
    return start, end


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


# ---------------------------------------------------------------------------
# Django template tag scanning
# ---------------------------------------------------------------------------


class _TagCtx:
    """The ``{% TAG ... %}`` enclosing the cursor.

    *name* — the tag name (``url``, ``block``, …).
    *body_start* — first character past ``{% TAG`` and any whitespace.
    *body_end* — offset of the closing ``%}`` (exclusive).
    """
    __slots__ = ("name", "body_start", "body_end")

    def __init__(self, name: str, body_start: int, body_end: int) -> None:
        self.name = name
        self.body_start = body_start
        self.body_end = body_end


_TAG_OPEN_RE = re.compile(r"\{%-?\s*([A-Za-z_][A-Za-z0-9_]*)")


def _enclosing_template_tag(source: str, offset: int) -> _TagCtx | None:
    """Return the ``{% ... %}`` tag enclosing *offset*, or None.

    A tag is enclosing when ``{%`` appears before the cursor and the
    matching ``%}`` either appears after the cursor or is missing
    entirely (the user is mid-edit). We tolerate unclosed tags so
    completion fires on the in-progress ``{% url '|``.
    """
    open_idx = source.rfind("{%", 0, offset)
    if open_idx < 0:
        return None
    # If a ``%}`` sits between the open and the cursor, the cursor is
    # outside the tag.
    close_between = source.find("%}", open_idx, offset)
    if close_between >= 0:
        return None
    m = _TAG_OPEN_RE.match(source, open_idx)
    if m is None:
        return None
    tag_name = m.group(1)
    body_start = m.end()
    # Skip the leading whitespace after the tag name so partials line up.
    while body_start < len(source) and source[body_start] in " \t":
        body_start += 1
    close_after = source.find("%}", offset)
    if close_after < 0:
        # Unclosed tag — assume body extends to end of line. Bounding it
        # avoids accidentally swallowing the rest of the file when the
        # user is mid-edit.
        nl = source.find("\n", offset)
        body_end = nl if nl >= 0 else len(source)
    else:
        body_end = close_after
    if offset < body_start or offset > body_end:
        return None
    return _TagCtx(name=tag_name, body_start=body_start, body_end=body_end)


def _enclosing_template_var(source: str, offset: int) -> tuple[int, int] | None:
    """Return ``(body_start, body_end)`` if *offset* is inside ``{{ … }}``.

    Mirrors :func:`_enclosing_template_tag` but for variable expressions
    (``{{ x|filter }}``). Tolerates an unclosed ``}}`` for the in-progress
    edit case — the body is then bounded to the end of the line.
    """
    open_idx = source.rfind("{{", 0, offset)
    if open_idx < 0:
        return None
    # If a ``}}`` sits between the open and the cursor, we're outside.
    close_between = source.find("}}", open_idx, offset)
    if close_between >= 0:
        return None
    body_start = open_idx + 2
    while body_start < len(source) and source[body_start] in " \t":
        body_start += 1
    close_after = source.find("}}", offset)
    if close_after < 0:
        nl = source.find("\n", offset)
        body_end = nl if nl >= 0 else len(source)
    else:
        body_end = close_after
    if offset < body_start or offset > body_end:
        return None
    return body_start, body_end


def _filter_partial_range(
    source: str, offset: int, body_start: int, body_end: int,
) -> tuple[int, int] | None:
    """If *offset* sits at a filter-name position, return the partial's range.

    A filter position is an identifier whose immediately-preceding
    non-identifier character is ``|``. The cursor may be at the end of an
    in-progress partial (``{{ x|tr|`` ← cursor) or just past the pipe
    with no partial typed yet (``{{ x||`` ← cursor). Returns
    ``(start, end)`` of the bare-word — possibly zero-width at *offset*.
    Returns ``None`` for any other position.
    """
    if offset < body_start or offset > body_end:
        return None
    # Don't fire inside a string literal — ``{{ x|default:"|fallback" }}``
    # has a pipe in a string that isn't a filter boundary.
    if _string_state_at_in_range(source, offset, body_start, body_end) is not None:
        return None
    start = offset
    while start > body_start and _is_word_char(source[start - 1]):
        start -= 1
    if start == body_start:
        return None
    if source[start - 1] != "|":
        return None
    end = offset
    while end < body_end and _is_word_char(source[end]):
        end += 1
    return start, end


_LOAD_RE = re.compile(r"\{%-?\s*load\s+([^%]*?)(?=-?%\})")


def _loaded_libraries(source: str) -> set[str]:
    """Return every library name referenced in a ``{% load … %}`` tag.

    Handles both forms — ``{% load lib1 lib2 %}`` (all listed names are
    libraries) and ``{% load tag1 tag2 from lib %}`` (only the name after
    ``from`` is the library; the others are individual tags being pulled
    in). For the second form we still index the library so completion
    can offer the full surface — pruning to the explicit ``from`` list
    would be more correct but trades off accuracy for a rare authoring
    pattern.
    """
    out: set[str] = set()
    for m in _LOAD_RE.finditer(source):
        body = m.group(1).strip()
        if not body:
            continue
        parts = body.split()
        if "from" in parts:
            idx = parts.index("from")
            if idx + 1 < len(parts):
                out.add(parts[idx + 1])
        else:
            out.update(parts)
    return out


_BLOCK_RE = re.compile(r"\{%-?\s*block\s+([A-Za-z_][A-Za-z0-9_]*)")
_EXTENDS_RE = re.compile(
    r"""\{%-?\s*extends\s+(?:'([^'\n]+)'|"([^"\n]+)")"""
)


def _block_names_in(text: str) -> set[str]:
    return {m.group(1) for m in _BLOCK_RE.finditer(text)}


def _extends_target(text: str) -> str | None:
    m = _EXTENDS_RE.search(text)
    if m is None:
        return None
    return m.group(1) or m.group(2)


# ---------------------------------------------------------------------------
# Unloaded-tag scanning
# ---------------------------------------------------------------------------


# Every ``{% tagname`` occurrence — the unloaded-tag scanner's raw feed.
_ANY_TAG_RE = re.compile(r"\{%-?\s*([A-Za-z_][A-Za-z0-9_]*)")

# Every ``{% ... %}`` / ``{{ ... }}`` construct. Mirrors Django's own
# lexer regex (``tag_re``), which is not DOTALL — constructs can't span
# newlines.
_CONSTRUCT_RE = re.compile(r"\{%.*?%\}|\{\{.*?\}\}")


def _filter_uses(source: str):
    """Yield ``(name, offset)`` for each ``|name`` filter use in *source*.

    Scans only inside ``{{ … }}`` / ``{% … %}`` constructs and tracks
    string-literal state so the pipe in ``{{ x|default:"a|b" }}`` doesn't
    produce a phantom ``b`` filter.
    """
    for m in _CONSTRUCT_RE.finditer(source):
        body = m.group(0)
        base = m.start()
        in_string: str | None = None
        i = 0
        n = len(body)
        while i < n:
            ch = body[i]
            if in_string is not None:
                if ch == "\\" and i + 1 < n:
                    i += 2
                    continue
                if ch == in_string:
                    in_string = None
                i += 1
                continue
            if ch in "\"'":
                in_string = ch
                i += 1
                continue
            if ch == "|":
                start = i + 1
                j = start
                while j < n and _is_word_char(body[j]):
                    j += 1
                if j > start and (body[start].isalpha() or body[start] == "_"):
                    yield body[start:j], base + start
                i = j
                continue
            i += 1


def _diag_range(source: str, start_offset: int, end_offset: int) -> dict:
    """Build an LSP range for a same-line span given by ``str`` offsets."""
    line_start = source.rfind("\n", 0, start_offset) + 1
    line_no = source.count("\n", 0, start_offset)
    return {
        "start": {
            "line": line_no,
            "character": _lsp_character_in_line(source, line_start, start_offset),
        },
        "end": {
            "line": line_no,
            "character": _lsp_character_in_line(source, line_start, end_offset),
        },
    }

_VERBATIM_RE = re.compile(
    r"\{%-?\s*verbatim(?:\s[^%]*)?-?%\}.*?\{%-?\s*endverbatim\s*-?%\}",
    re.DOTALL,
)
_COMMENT_RE = re.compile(
    r"\{%-?\s*comment(?:\s[^%]*)?-?%\}.*?\{%-?\s*endcomment\s*-?%\}",
    re.DOTALL,
)


def _excluded_spans(source: str) -> list[tuple[int, int]]:
    """Spans where ``{% ... %}`` content isn't parsed as a tag.

    ``{% verbatim %}`` and ``{% comment %}`` bodies — tag-looking text
    inside them never reaches Django's tag parser. HTML comments are
    *not* excluded: Django parses tags inside ``<!-- -->`` too, so an
    unloaded tag there still breaks the render.
    """
    spans: list[tuple[int, int]] = []
    for rx in (_VERBATIM_RE, _COMMENT_RE):
        spans.extend(m.span() for m in rx.finditer(source))
    return spans


def _lsp_position_at(source: str, offset: int) -> dict:
    line = source.count("\n", 0, offset)
    line_start = source.rfind("\n", 0, offset) + 1
    return {
        "line": line,
        "character": _lsp_character_in_line(source, line_start, offset),
    }


def _pos_tuple(p: dict) -> tuple[int, int]:
    return int(p.get("line", 0)), int(p.get("character", 0))


def _ranges_overlap(a: dict, b: dict) -> bool:
    """Inclusive overlap — a zero-width cursor range touching either
    end of the diagnostic still counts (that's what editors send when
    the caret merely sits on the squiggle)."""
    return (
        _pos_tuple(a["start"]) <= _pos_tuple(b["end"])
        and _pos_tuple(b["start"]) <= _pos_tuple(a["end"])
    )


def _load_fix_edit(source: str, lib: str) -> tuple[dict, bool]:
    """Build the TextEdit that loads *lib*; returns ``(edit, appended)``.

    *appended* is True when the edit extends an existing simple
    ``{% load a b %}`` tag (the ``from`` form gets a fresh line instead
    — appending a library name to ``{% load x from y %}`` would change
    its meaning). Otherwise a new ``{% load lib %}`` line is inserted
    after ``{% extends %}`` when present (Django requires extends to be
    the template's first tag), else at the very top of the file.
    """
    for m in _LOAD_RE.finditer(source):
        body = m.group(1)
        if "from" in body.split():
            continue
        insert_at = m.start(1) + len(body.rstrip())
        pos = _lsp_position_at(source, insert_at)
        return {"range": {"start": pos, "end": pos}, "newText": f" {lib}"}, True

    m = _EXTENDS_RE.search(source)
    if m is not None:
        nl = source.find("\n", m.end())
        if nl >= 0:
            pos = _lsp_position_at(source, nl + 1)
            return {"range": {"start": pos, "end": pos}, "newText": f"{{% load {lib} %}}\n"}, False
        pos = _lsp_position_at(source, len(source))
        return {"range": {"start": pos, "end": pos}, "newText": f"\n{{% load {lib} %}}\n"}, False

    pos = {"line": 0, "character": 0}
    return {"range": {"start": pos, "end": pos}, "newText": f"{{% load {lib} %}}\n"}, False


# ---------------------------------------------------------------------------
# {% url 'name' %} diagnostics
# ---------------------------------------------------------------------------


_URL_TAG_RE = re.compile(
    r"""\{%-?\s*url\s+(?:'([^'\n]*)'|"([^"\n]*)")""",
    re.MULTILINE,
)


def _template_url_diagnostics(source: str, url_index):
    """Yield diagnostics for ``{% url 'unknown' %}`` references."""
    for m in _URL_TAG_RE.finditer(source):
        value = m.group(1) if m.group(1) is not None else m.group(2)
        if not value:
            continue
        # The value may use Django's namespace syntax — already handled by
        # the index (entries are stored as ``ns:name``). A lookup is enough.
        if value in url_index.entries:
            continue
        # Skip dynamic-looking values: variables in templates aren't
        # quoted, so anything in this regex is already a literal — no
        # further heuristic needed.
        # Locate the value's span for the diagnostic range.
        # Group 1 covers the single-quoted form, group 2 the double-quoted
        # form; whichever matched, ``m.start(group)`` points at the first
        # char after the opening quote.
        group_idx = 1 if m.group(1) is not None else 2
        start_offset = m.start(group_idx)
        end_offset = m.end(group_idx)
        line_start = source.rfind("\n", 0, start_offset) + 1
        line_no = source.count("\n", 0, start_offset)
        col_start = _lsp_character_in_line(source, line_start, start_offset)
        col_end = _lsp_character_in_line(source, line_start, end_offset)
        msg = f"unknown URL name {value!r} (referenced via {{% url %}})"
        yield {
            "code": URL_DIAG_CODE,
            "message": msg,
            "range": {
                "start": {"line": line_no, "character": col_start},
                "end": {"line": line_no, "character": col_end},
            },
            "severity": 2,
            "source": DIAG_SOURCE,
            "data": {"value": value, "callee": "url-tag"},
        }
