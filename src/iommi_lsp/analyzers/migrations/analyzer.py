"""Migration awareness — autocomplete for ``Migration.dependencies`` tuples.

Walks every ``<app>/migrations/*.py`` under the workspace (excluding
``__init__.py``) and groups the migration filenames by their parent
package (which we treat as the app label — matches Django's convention
that ``app/migrations/0001_initial.py`` has ``"app"`` as its label).

When the cursor sits inside the second string of a ``(app, migration)``
tuple under ``dependencies``, we offer that app's migration names.

Discovery is one-shot at index time. New migration files created during
an editing session aren't visible until restart.
"""

from __future__ import annotations

import ast
import os
import re
from collections.abc import Callable
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse

from ... import log
from ..base import CompletionResult, Diagnostic


_log = log.get("migrations.analyzer")


_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "build", "dist", "site-packages",
})


def discover_migrations(workspace_root: Path) -> dict[str, list[str]]:
    """Return ``{app_label: [migration_name, …]}`` for every workspace app.

    *migration_name* is the file basename with ``.py`` stripped. Sorted
    so the most recent (highest-numbered) name surfaces predictably.
    """
    out: dict[str, list[str]] = defaultdict(list)
    root = workspace_root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        if os.path.basename(dirpath) != "migrations":
            continue
        # Skip non-package migration folders (must contain __init__.py).
        if "__init__.py" not in filenames:
            continue
        app_label = os.path.basename(os.path.dirname(dirpath))
        if not app_label:
            continue
        for name in filenames:
            if not name.endswith(".py") or name == "__init__.py" or name.startswith("."):
                continue
            out[app_label].append(name[:-3])
    return {k: sorted(v) for k, v in out.items()}


class MigrationsAnalyzer:
    """Implements the :class:`Analyzer` Protocol for migrations completion."""

    name = "migrations"

    def __init__(
        self,
        workspace_root: Path,
        text_provider: Callable[[str], str | None] | None = None,
        parse_provider: "Callable[[str, str], ast.Module | None] | None" = None,
    ) -> None:
        self.workspace_root = workspace_root
        self._text_provider = text_provider
        self._parse_provider = parse_provider
        self._migrations: dict[str, list[str]] = {}

    @property
    def migrations(self) -> dict[str, list[str]]:
        return dict(self._migrations)

    async def index(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self._migrations = discover_migrations(workspace_root)
        _log.info(
            "indexed migrations: %s",
            {k: len(v) for k, v in self._migrations.items()},
        )

    async def on_file_changed(self, uri: str) -> None:
        return None

    def is_false_positive(self, uri: str, diagnostic: Diagnostic) -> bool:
        if not _is_unresolved_attribute(diagnostic):
            return False
        path = _uri_to_path(uri)
        if path is None:
            return False
        # ``RunPython.noop`` / ``RunSQL.noop`` only appears inside
        # ``<app>/migrations/<NNNN>_*.py``. Gate on that before doing
        # any per-file work — without this, the analyzer would parse the
        # buffer once per diagnostic on every file the user opens.
        if not _looks_like_migration_path(path):
            return False
        source = self._source_for(uri, path)
        if source is None:
            return False
        tree = self._parse(uri, source)
        if tree is None:
            return False
        try:
            return _is_migration_noop_attr(tree, diagnostic)
        except Exception:
            _log.exception(
                "migration noop suppression check crashed; keeping the diagnostic"
            )
            return False

    def _parse(self, uri: str, source: str) -> ast.Module | None:
        if self._parse_provider is not None:
            return self._parse_provider(uri, source)
        try:
            return ast.parse(source)
        except SyntaxError:
            return None

    def additional_diagnostics(self, uri: str) -> list[Diagnostic]:
        return []

    def completions(self, uri: str, position: dict) -> CompletionResult:
        empty = CompletionResult()
        if not self._migrations:
            return empty
        path = _uri_to_path(uri)
        if path is None:
            return empty
        source = self._source_for(uri, path)
        if source is None:
            return empty
        try:
            return _scan_completions(source, position, self._migrations)
        except Exception:
            _log.exception("migrations completion scanner crashed; emitting nothing")
            return empty

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
# False-positive suppression for ``RunPython.noop`` / ``RunSQL.noop``
# ---------------------------------------------------------------------------


_NOOP_OWNERS: frozenset[str] = frozenset({"RunPython", "RunSQL"})


def _is_unresolved_attribute(diagnostic: Diagnostic) -> bool:
    code = diagnostic.get("code")
    if isinstance(code, str) and code == "unresolved-attribute":
        return True
    if isinstance(code, dict) and code.get("value") == "unresolved-attribute":
        return True
    return False


_MIGRATION_FILENAME_RE = re.compile(r"^\d{4}_.*\.py$")


def _looks_like_migration_path(path: Path) -> bool:
    """Heuristic gate for ``is_false_positive`` — skip non-migration files.

    Real Django migrations live in ``<app>/migrations/<NNNN>_*.py``, but
    we accept either the directory or the filename pattern so the
    analyzer keeps working in synthetic test fixtures where the file
    isn't necessarily nested under ``migrations/``.
    """
    parts = path.parts
    if len(parts) >= 2 and parts[-2] == "migrations":
        return True
    return bool(_MIGRATION_FILENAME_RE.match(path.name))


def _is_migration_noop_attr(tree: ast.Module, diagnostic: Diagnostic) -> bool:
    """``RunPython.noop`` / ``RunSQL.noop`` / ``migrations.RunPython.noop``
    — Django attaches ``noop`` as a class attribute on these operation
    classes. ty can't see it without runtime stubs; drop the warning.
    """
    rng = diagnostic.get("range") or {}
    attr = _find_attribute_at(tree, rng)
    if attr is None or attr.attr != "noop":
        return False
    receiver = attr.value
    if isinstance(receiver, ast.Name):
        return receiver.id in _NOOP_OWNERS
    if isinstance(receiver, ast.Attribute):
        return receiver.attr in _NOOP_OWNERS
    return False


_ATTR_INDEX_ATTR = "_iommi_lsp_migrations_attr_index"


def _attr_index(tree: ast.Module) -> list[ast.Attribute]:
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


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------


_MARKER = "__iommi_lsp_migrations_marker__"


def _scan_completions(
    source: str, position: dict, migrations: dict[str, list[str]],
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

    info = _locate_marker_in_dependency_tuple(tree)
    if info is None:
        return empty
    app_label = info
    names = migrations.get(app_label)
    if not names:
        return empty

    partial = source[ctx.start + 1: offset]
    line_start = source.rfind("\n", 0, offset) + 1
    start_character = _lsp_character_in_line(source, line_start, ctx.start + 1)
    edit_range = {
        "start": {"line": line, "character": start_character},
        "end": {"line": line, "character": character},
    }

    items: list[dict] = []
    for name in names:
        if partial and not name.startswith(partial):
            continue
        items.append({
            "label": name,
            "kind": 21,   # Constant
            "insertText": name,
            "textEdit": {"range": edit_range, "newText": name},
            "detail": f"migration ({app_label})",
            "data": {"source": "iommi_lsp.migration-name", "app": app_label},
        })
    return CompletionResult(items=items, exclusive=True)


def _locate_marker_in_dependency_tuple(tree: ast.Module) -> str | None:
    """Find the sentinel and return the app label if it sits in a dep tuple.

    Recognised shape: ``dependencies = [('app', '‸'), ...]`` inside a
    ``class Migration(migrations.Migration):``. We don't enforce the
    class-base constraint strictly — any ``dependencies =`` whose value
    is a list of ``(app, name)`` tuples is treated the same way.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Tuple) or len(node.elts) != 2:
            continue
        first, second = node.elts
        if not (
            isinstance(second, ast.Constant)
            and second.value == _MARKER
        ):
            continue
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        # Verify enclosing assignment is named ``dependencies``.
        if _enclosed_in_dependencies(tree, node):
            return first.value
    return None


def _enclosed_in_dependencies(tree: ast.Module, target: ast.AST) -> bool:
    """Walk the tree and check whether *target* lives under an
    assignment whose target is named ``dependencies``."""
    for stmt in ast.walk(tree):
        if not isinstance(stmt, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "dependencies"
            for t in stmt.targets
        ):
            continue
        for sub in ast.walk(stmt.value):
            if sub is target:
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
