"""Template-name completion inside string literals.

Walks the workspace at init time and collects every file under a
``templates/`` directory — Django's app-templates convention. Each
template name is the file path relative to its enclosing
``templates/`` dir, so ``myapp/templates/myapp/index.html`` becomes
``myapp/index.html``.

At completion time, when the cursor sits inside a single-line string
literal whose pre-cursor content already contains a ``/``, we offer
every known template that starts with that prefix. The ``/`` trigger
is a cheap heuristic — random strings rarely contain a slash, but
Django/iommi template references almost always do.

Discovery is one-shot at ``index`` time. New template files created
after the LSP starts are not visible until restart (deliberate — keeps
the indexer dead simple; the user mentions this is fine for now).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote, urlparse

from ... import log
from ..base import CompletionResult, Diagnostic


_log = log.get("templates.analyzer")


_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "build", "dist", "site-packages",
})


class TemplateAnalyzer:
    """Implements the :class:`Analyzer` Protocol for template-name completion."""

    name = "templates"

    def __init__(
        self,
        workspace_root: Path,
        text_provider: Callable[[str], str | None] | None = None,
    ) -> None:
        self.workspace_root = workspace_root
        self._text_provider = text_provider
        self._templates: list[str] = []

    @property
    def templates(self) -> list[str]:
        return list(self._templates)

    # -- Analyzer protocol ----------------------------------------------------

    async def index(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self._templates = sorted(discover_templates(workspace_root))
        _log.info(
            "indexed %d templates under %s", len(self._templates), workspace_root,
        )

    async def on_file_changed(self, uri: str) -> None:
        # One-shot discovery — new templates aren't picked up until the
        # LSP restarts. Documented behaviour; revisit when users hit it.
        return None

    def is_false_positive(self, uri: str, diagnostic: Diagnostic) -> bool:
        return False

    def additional_diagnostics(self, uri: str) -> list[Diagnostic]:
        return []

    def completions(self, uri: str, position: dict) -> CompletionResult:
        empty = CompletionResult()
        if not self._templates:
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
            _log.exception("template completion scanner crashed; emitting nothing")
            return empty

    # -- internals ------------------------------------------------------------

    def _scan_completions(self, source: str, position: dict) -> CompletionResult:
        empty = CompletionResult()
        line = int(position.get("line", 0))
        character = int(position.get("character", 0))
        offset = _offset_from_lsp_position(source, line, character)
        if offset > len(source):
            return empty

        ctx = _string_state_at(source, offset)
        if ctx is None:
            return empty

        partial = source[ctx.start + 1: offset]
        if "/" not in partial:
            return empty

        # An explicit replacement range is the only way to keep editors
        # that treat `/` as a word boundary (Helix, Neovim's built-in
        # client) from replacing only the trailing word — without it,
        # accepting ``reviews/reviews__tags.html`` on the partial
        # ``reviews/rev`` produces ``reviews/reviews/reviews__tags.html``.
        line_start = source.rfind("\n", 0, offset) + 1
        start_character = _lsp_character_in_line(source, line_start, ctx.start + 1)
        edit_range = {
            "start": {"line": line, "character": start_character},
            "end": {"line": line, "character": character},
        }

        items: list[dict] = []
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
# Helpers
# ---------------------------------------------------------------------------


def discover_templates(workspace_root: Path) -> set[str]:
    """Return every template name under any ``templates/`` directory.

    A *template name* is the file path relative to the enclosing
    ``templates/`` directory, in POSIX form. Dotfiles and hidden
    directories are skipped; standard noise dirs (``.venv``, ``build``,
    …) are pruned from the search.
    """
    out: set[str] = set()
    root = workspace_root.resolve()
    templates_dirs: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        if "templates" in dirnames:
            templates_dirs.append(Path(dirpath) / "templates")
            # Don't double-descend — we walk each templates dir below.
            dirnames.remove("templates")
    for tdir in templates_dirs:
        for sub_dir, sub_dirnames, sub_files in os.walk(tdir):
            sub_dirnames[:] = [d for d in sub_dirnames if not d.startswith(".")]
            for name in sub_files:
                if name.startswith("."):
                    continue
                rel = Path(sub_dir, name).relative_to(tdir)
                out.add(rel.as_posix())
    return out


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
