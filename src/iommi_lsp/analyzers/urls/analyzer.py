"""URL-name index + completion + diagnostics.

Walks the workspace's ``urls.py`` files at index time and builds a list
of registered URL names. ``path(...)`` / ``re_path(...)`` / ``url(...)``
calls with a ``name='...'`` kwarg contribute one entry each. Namespaces
are tracked via two routes:

* ``include('app.urls', namespace='ns')`` — the include call carries the
  namespace explicitly.
* A module-level ``app_name = 'ns'`` in the included urls module — Django
  honours this when the include call passes the included module without
  a ``namespace=`` override.

At completion time, when the cursor sits inside a string literal that is
the first argument to ``reverse(...)``, ``redirect(...)``, or
``resolve_url(...)``, we offer the known names with optional ``ns:`` prefix.

At diagnostic time we emit ``django-unknown-url-name`` for strings that
don't match any registered name. Bias is toward false negatives — if we
can't read the file or there are zero names registered, we say nothing.

Templates are not currently scanned for ``{% url 'name' %}`` validation
(no template AST integration); the existing iommi/templates analyzers
already touch template paths, but ``{% url %}`` arg validation would
require a small string-token scanner of its own.
"""

from __future__ import annotations

import ast
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

from ... import log
from ..base import CompletionResult, Diagnostic


_log = log.get("urls.analyzer")


_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "build", "dist", "site-packages",
})


# Callable names whose first positional/``viewname=`` argument is a URL name.
_REVERSE_CALL_NAMES: frozenset[str] = frozenset({
    "reverse", "reverse_lazy", "redirect", "resolve_url",
})


URL_DIAG_CODE = "django-unknown-url-name"
URL_DIAG_SOURCE = "iommi_lsp"


@dataclass
class UrlEntry:
    name: str                  # final name with namespace prefix (e.g. "blog:detail")
    file_path: Path
    line: int                  # 1-indexed line where the name= kwarg lives


@dataclass
class UrlIndex:
    """All URL names registered across the workspace.

    Names are stored both as a flat set (fast lookup) and with their
    file/line metadata (so we can report hits later if useful).
    """

    entries: dict[str, UrlEntry] = field(default_factory=dict)

    @property
    def names(self) -> set[str]:
        return set(self.entries.keys())

    def add(self, entry: UrlEntry) -> None:
        # Last-write-wins on duplicates. Real projects rarely re-register
        # the same name; when they do, exactly which file we point at
        # isn't important enough to deduplicate carefully.
        self.entries[entry.name] = entry


class UrlAnalyzer:
    """Implements the :class:`Analyzer` Protocol for URL-name awareness."""

    name = "urls"

    def __init__(
        self,
        workspace_root: Path,
        text_provider: Callable[[str], str | None] | None = None,
    ) -> None:
        self.workspace_root = workspace_root
        self._text_provider = text_provider
        self.url_index = UrlIndex()

    @property
    def names(self) -> set[str]:
        return self.url_index.names

    # -- Analyzer protocol ----------------------------------------------------

    async def index(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.url_index = build_url_index(workspace_root)
        _log.info(
            "indexed %d URL names under %s",
            len(self.url_index.entries), workspace_root,
        )

    async def on_file_changed(self, uri: str) -> None:
        # urls.py files don't change every keystroke; on the rare edit
        # we do a single re-scrape of just that file. Cheap enough.
        path = _uri_to_path(uri)
        if path is None:
            return
        if not _looks_like_urls_module(path):
            return
        # Drop entries from this file, then rescan.
        for n, e in list(self.url_index.entries.items()):
            if e.file_path == path:
                self.url_index.entries.pop(n, None)
        scrape = _scrape_urls_file(path)
        if scrape is None:
            return
        # Use a fresh resolver pass for just this file. Cross-file include
        # namespaces won't be re-checked but stale entries are at worst
        # cosmetic.
        for raw in scrape.entries:
            full = raw.name
            entry = UrlEntry(name=full, file_path=path, line=raw.line)
            self.url_index.add(entry)

    def is_false_positive(self, uri: str, diagnostic: Diagnostic) -> bool:
        return False

    def additional_diagnostics(self, uri: str) -> list[Diagnostic]:
        if not self.url_index.entries:
            return []
        path = _uri_to_path(uri)
        if path is None:
            return []
        source = self._source_for(uri, path)
        if source is None:
            return []
        try:
            return list(_scan_diagnostics(source, self.url_index))
        except Exception:
            _log.exception("url diagnostic scanner crashed; emitting nothing")
            return []

    def completions(self, uri: str, position: dict) -> CompletionResult:
        empty = CompletionResult()
        if not self.url_index.entries:
            return empty
        path = _uri_to_path(uri)
        if path is None:
            return empty
        source = self._source_for(uri, path)
        if source is None:
            return empty
        try:
            return _scan_completions(source, position, self.url_index)
        except Exception:
            _log.exception("url completion scanner crashed; emitting nothing")
            return empty

    # -- internals ------------------------------------------------------------

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
# Workspace discovery + indexing.
# ---------------------------------------------------------------------------


@dataclass
class _RawEntry:
    name: str
    line: int


@dataclass
class _UrlsFileScrape:
    """Per-file scrape of urls.py:

    * ``entries`` — names registered directly in this file (already
      prefixed with any locally-detected namespace from includes that
      named one explicitly, but *not* with the parent file's namespace —
      that's the resolver's job).
    * ``app_name`` — the module-level ``app_name = '…'`` if present.
    * ``includes`` — for each ``include(…)`` call, the (module_or_url,
      namespace) tuple. Used by the resolver to walk into included urls
      modules and propagate namespaces. ``namespace`` is None when not
      passed explicitly; the included module's ``app_name`` is the
      fallback.
    * ``file_path`` — for diagnostics.
    """

    file_path: Path
    entries: list[_RawEntry]
    app_name: str | None
    includes: list[tuple[str | None, str | None, int]]   # (module_target, namespace, line)


def _looks_like_urls_module(path: Path) -> bool:
    """A urls.py at any depth. We don't try to figure out which one is
    *the* root urls.py — we index names from all of them, which is what
    ``reverse()`` ends up needing anyway."""
    return path.name == "urls.py"


def discover_urls(workspace_root: Path) -> list[Path]:
    """Return every ``urls.py`` under *workspace_root*."""
    out: list[Path] = []
    root = workspace_root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        if "urls.py" in filenames:
            out.append(Path(dirpath) / "urls.py")
    return out


def _string_value(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _flat_attr(node: ast.AST) -> str | None:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _is_path_call(node: ast.Call) -> bool:
    """Recognise ``path(...)`` / ``re_path(...)`` / ``url(...)``."""
    func = node.func
    if isinstance(func, ast.Name) and func.id in {"path", "re_path", "url"}:
        return True
    if isinstance(func, ast.Attribute) and func.attr in {"path", "re_path", "url"}:
        return True
    return False


def _is_include_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id == "include":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "include":
        return True
    return False


def _module_app_name(tree: ast.Module) -> str | None:
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            tgt = stmt.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id == "app_name":
                v = _string_value(stmt.value)
                if v:
                    return v
        elif (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "app_name"
        ):
            v = _string_value(stmt.value)
            if v:
                return v
    return None


def _scrape_urls_file(path: Path) -> _UrlsFileScrape | None:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        _log.debug("could not read %s: %s", path, e)
        return None
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        _log.debug("could not parse %s: %s", path, e)
        return None

    app_name = _module_app_name(tree)
    entries: list[_RawEntry] = []
    includes: list[tuple[str | None, str | None, int]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _is_path_call(node):
            name = _name_kwarg(node)
            if name is not None:
                entries.append(_RawEntry(name=name, line=node.lineno))
            # path('foo/', include('app.urls', namespace='ns')) — also scan.
            for arg in node.args:
                if isinstance(arg, ast.Call) and _is_include_call(arg):
                    includes.append(_include_target(arg))
        elif _is_include_call(node):
            # Top-level include — rare but allowed.
            includes.append(_include_target(node))

    return _UrlsFileScrape(
        file_path=path,
        entries=entries,
        app_name=app_name,
        includes=includes,
    )


def _name_kwarg(call: ast.Call) -> str | None:
    for kw in call.keywords:
        if kw.arg == "name":
            v = _string_value(kw.value)
            if v:
                return v
    return None


def _include_target(call: ast.Call) -> tuple[str | None, str | None, int]:
    """Extract (module_dotted, namespace, line) from an include(...) call."""
    module: str | None = None
    namespace: str | None = None
    if call.args:
        arg0 = call.args[0]
        s = _string_value(arg0)
        if s is not None:
            module = s
        else:
            # Could be a tuple-form (urlpatterns, app_namespace) — we
            # don't try to follow non-string targets.
            flat = _flat_attr(arg0)
            if flat is not None:
                module = flat
    for kw in call.keywords:
        if kw.arg == "namespace":
            ns = _string_value(kw.value)
            if ns:
                namespace = ns
    return module, namespace, call.lineno


def _module_to_path(workspace_root: Path, module: str) -> Path | None:
    """Resolve a dotted module like ``blog.urls`` to a file path under root."""
    parts = module.split(".")
    if not parts:
        return None
    candidate = workspace_root / Path(*parts).with_suffix(".py")
    if candidate.is_file():
        return candidate
    candidate = workspace_root / Path(*parts) / "__init__.py"
    if candidate.is_file():
        return candidate
    return None


def build_url_index(workspace_root: Path) -> UrlIndex:
    """Scan every urls.py under *workspace_root* and build the name index.

    Includes are walked once per file so namespaces propagate from
    parent → child correctly. We don't perform a true recursive resolve
    (no fixed-point loop) — Django's URL configurations are trees in
    practice, and a single pass suffices when each include either
    overrides the namespace explicitly or the included module declares
    its own ``app_name``.
    """
    workspace_root = workspace_root.resolve()
    scrapes: dict[Path, _UrlsFileScrape] = {}
    for path in discover_urls(workspace_root):
        scrape = _scrape_urls_file(path)
        if scrape is not None:
            scrapes[path] = scrape

    index = UrlIndex()
    # First pass: register every name as-is (no namespace prefix). This
    # gives us coverage even when include resolution fails.
    for path, scrape in scrapes.items():
        for raw in scrape.entries:
            index.add(UrlEntry(name=raw.name, file_path=path, line=raw.line))

    # Second pass: resolve include namespaces. For each include that
    # points at a workspace urls module, prefix that module's entries
    # with the effective namespace.
    for path, scrape in scrapes.items():
        for module, namespace, _line in scrape.includes:
            if module is None:
                continue
            target_path = _module_to_path(workspace_root, module)
            if target_path is None or target_path not in scrapes:
                continue
            target_scrape = scrapes[target_path]
            effective_ns = namespace or target_scrape.app_name
            if not effective_ns:
                continue
            for raw in target_scrape.entries:
                full = f"{effective_ns}:{raw.name}"
                index.add(UrlEntry(
                    name=full, file_path=target_path, line=raw.line,
                ))
    return index


# ---------------------------------------------------------------------------
# Completion + diagnostics at the use site.
# ---------------------------------------------------------------------------


def _scan_completions(
    source: str, position: dict, index: UrlIndex,
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

    call = _enclosing_reverse_call(source, ctx.start)
    if call is None:
        return empty

    partial = source[ctx.start + 1: offset]
    line_start = source.rfind("\n", 0, offset) + 1
    start_character = _lsp_character_in_line(source, line_start, ctx.start + 1)
    edit_range = {
        "start": {"line": line, "character": start_character},
        "end": {"line": line, "character": character},
    }

    items: list[dict] = []
    for name in sorted(index.entries):
        if partial and not name.startswith(partial):
            continue
        items.append({
            "label": name,
            "kind": 21,   # Constant
            "insertText": name,
            "textEdit": {"range": edit_range, "newText": name},
            "detail": f"URL name ({call})",
            "data": {"source": "iommi_lsp.url-name"},
        })
    if not items and partial:
        # Recognised position with no matches — still exclusive so ty's
        # free-form name list doesn't muscle in with random workspace
        # variables.
        return CompletionResult(items=[], exclusive=True)
    return CompletionResult(items=items, exclusive=True)


def _scan_diagnostics(source: str, index: UrlIndex):
    """Yield diagnostics for ``reverse('unknown')`` style calls."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = _callee_simple_name(node.func)
        if callee not in _REVERSE_CALL_NAMES:
            continue
        # First positional or ``viewname=`` / ``to=`` kwarg.
        target = _first_string_target(node)
        if target is None:
            continue
        const_node, value = target
        if value in index.entries:
            continue
        # Heuristic: redirect() may receive a path (e.g. "/foo/"), a
        # full URL, or a relative path ("." / ".."); treat any of those
        # as non-name. Same for ``resolve_url``.
        if callee in {"redirect", "resolve_url"} and (
            "/" in value or value in {".", ".."}
        ):
            continue
        # Skip empty strings — common during typing.
        if not value:
            continue
        diag = _make_diagnostic(const_node, value, callee, index)
        if diag is not None:
            yield diag


def _callee_simple_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _first_string_target(call: ast.Call) -> tuple[ast.Constant, str] | None:
    """Return (node, value) for the URL-name argument, if any."""
    # ``reverse(viewname=…)`` is the canonical kwarg; ``redirect(to=…)``
    # is rare but allowed. Positional first arg covers the common case.
    if call.args:
        a0 = call.args[0]
        if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
            return a0, a0.value
        # Not a literal — bail (could be a variable).
        return None
    for kw in call.keywords:
        if kw.arg in {"viewname", "to"}:
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return kw.value, kw.value.value
    return None


def _make_diagnostic(
    const_node: ast.Constant, value: str, callee: str, index: UrlIndex,
) -> Diagnostic | None:
    if const_node.lineno is None or const_node.col_offset is None:
        return None
    line0 = const_node.lineno - 1
    col_start = const_node.col_offset + 1   # skip opening quote
    col_end = col_start + len(value)
    msg = f"unknown URL name {value!r} (called via {callee}())"
    # Surface close matches when small (Levenshtein-ish prefix-based hint).
    hints = _close_matches(value, index.entries.keys())
    if hints:
        msg += f"  (did you mean: {', '.join(hints[:3])}?)"
    return {
        "code": URL_DIAG_CODE,
        "message": msg,
        "range": {
            "start": {"line": line0, "character": col_start},
            "end": {"line": line0, "character": col_end},
        },
        "severity": 2,
        "source": URL_DIAG_SOURCE,
        "data": {"value": value, "callee": callee},
    }


def _close_matches(needle: str, names) -> list[str]:
    """Tiny suggestion helper — names that share a prefix or contain *needle*.

    Not a full edit-distance calculation: a substring scan is enough to
    catch typos like ``detial`` vs ``detail`` if we lower the bar to
    "shares 60% characters in order". Keep it simple — overly aggressive
    suggestions are worse than none. Returned list is empty when the
    needle is too short to match meaningfully.
    """
    if len(needle) < 3:
        return []
    needle_lc = needle.lower()
    out: list[str] = []
    for n in names:
        nl = n.lower()
        # Strip namespace prefix for comparison.
        base = nl.rsplit(":", 1)[-1]
        if base.startswith(needle_lc[:3]) or needle_lc[:3] in base:
            out.append(n)
    return sorted(out)[:5]


# ---------------------------------------------------------------------------
# Recognising ``reverse('‸')`` etc. — the call site lookup.
# ---------------------------------------------------------------------------


def _enclosing_reverse_call(source: str, string_start: int) -> str | None:
    """Return the callee name if the open string at *string_start* is the
    first arg of a recognised reverse-style call. Returns ``None`` if not.

    Cheap scan — walks back over the immediately preceding non-whitespace
    text looking for ``(`` and then an identifier. Anything more
    complicated would require an AST parse with an open-string sentinel.
    """
    # Find the opening paren immediately before the string (allowing
    # whitespace).
    i = string_start - 1
    while i >= 0 and source[i] in " \t\r\n":
        i -= 1
    if i < 0 or source[i] != "(":
        return None
    j = i - 1
    end = j + 1
    while j >= 0 and (source[j].isalnum() or source[j] == "_"):
        j -= 1
    name = source[j + 1:end]
    if not name:
        return None
    if name not in _REVERSE_CALL_NAMES:
        return None
    return name


# ---------------------------------------------------------------------------
# Generic helpers (mirror of other analyzers' lightweight LSP utilities).
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


def _string_state_at(source: str, offset: int) -> _StringCtx | None:
    """Return the open single-line string at *offset*, or ``None``."""
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
            return None
        i += 1
    if in_string is None or string_start_in_line < 0:
        return None
    return _StringCtx(quote=in_string, start=line_start + string_start_in_line)
