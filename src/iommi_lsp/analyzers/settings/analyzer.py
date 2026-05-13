"""Completion for dotted-path string values in Django settings.

When the cursor sits inside an open single-line string literal that's
syntactically the value of one of a handful of recognised Django
settings (``INSTALLED_APPS``, ``MIDDLEWARE``, ``AUTHENTICATION_BACKENDS``,
``DEFAULT_AUTO_FIELD``, ``AUTH_USER_MODEL``, ``WSGI_APPLICATION``,
``AUTH_PASSWORD_VALIDATORS``, ``DEFAULT_EXCEPTION_REPORTER``), we offer
the appropriate set of completions:

* ``INSTALLED_APPS`` — Django's ``django.contrib.*`` apps plus any
  workspace package containing an ``apps.py``.
* ``MIDDLEWARE`` — built-in Django middleware classes.
* ``AUTHENTICATION_BACKENDS`` — built-in auth backends.
* ``DEFAULT_AUTO_FIELD`` — built-in ``AutoField`` classes.
* ``AUTH_USER_MODEL`` — ``auth.User`` plus workspace models in
  ``app_label.ModelName`` form (drawn from the Django index).
* ``WSGI_APPLICATION`` — workspace ``wsgi.py`` ``application`` exports.
* ``AUTH_PASSWORD_VALIDATORS`` — Django's password validators (only
  when the cursor is inside the ``NAME`` value of a dict entry).
* ``DEFAULT_EXCEPTION_REPORTER`` — built-in exception reporters.

Detection is AST-based: we replace the open string with a known
sentinel constant and walk the parse tree to find what setting (and
sub-context, for ``AUTH_PASSWORD_VALIDATORS``) the cursor is in. This
sidesteps the brittle "scan back through brackets" approach and
naturally ignores irrelevant string positions.
"""

from __future__ import annotations

import ast
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

from ... import log
from ..base import CompletionResult, Diagnostic


if TYPE_CHECKING:
    from ..django.index import DjangoIndex


_log = log.get("settings.analyzer")


_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "build", "dist", "site-packages",
})


# ---------------------------------------------------------------------------
# Built-in Django values per setting.
# ---------------------------------------------------------------------------


DJANGO_CONTRIB_APPS: tuple[str, ...] = (
    "django.contrib.admin",
    "django.contrib.admindocs",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.flatpages",
    "django.contrib.gis",
    "django.contrib.humanize",
    "django.contrib.messages",
    "django.contrib.postgres",
    "django.contrib.redirects",
    "django.contrib.sessions",
    "django.contrib.sitemaps",
    "django.contrib.sites",
    "django.contrib.staticfiles",
    "django.contrib.syndication",
)

# iommi ships its own AppConfig; surface it as a completion since
# iommi_lsp users are by definition iommi users.
EXTRA_APPS: tuple[str, ...] = (
    "iommi",
)


DJANGO_MIDDLEWARE: tuple[str, ...] = (
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.auth.middleware.RemoteUserMiddleware",
    "django.contrib.auth.middleware.PersistentRemoteUserMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "django.middleware.gzip.GZipMiddleware",
    "django.middleware.locale.LocaleMiddleware",
    "django.middleware.http.ConditionalGetMiddleware",
    "django.middleware.cache.UpdateCacheMiddleware",
    "django.middleware.cache.FetchFromCacheMiddleware",
    "django.contrib.sites.middleware.CurrentSiteMiddleware",
    "django.contrib.flatpages.middleware.FlatpageFallbackMiddleware",
    "django.contrib.redirects.middleware.RedirectFallbackMiddleware",
)


DJANGO_AUTH_BACKENDS: tuple[str, ...] = (
    "django.contrib.auth.backends.ModelBackend",
    "django.contrib.auth.backends.AllowAllUsersModelBackend",
    "django.contrib.auth.backends.RemoteUserBackend",
    "django.contrib.auth.backends.AllowAllUsersRemoteUserBackend",
)


DJANGO_AUTO_FIELDS: tuple[str, ...] = (
    "django.db.models.AutoField",
    "django.db.models.BigAutoField",
    "django.db.models.SmallAutoField",
)


DJANGO_PASSWORD_VALIDATORS: tuple[str, ...] = (
    "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    "django.contrib.auth.password_validation.MinimumLengthValidator",
    "django.contrib.auth.password_validation.CommonPasswordValidator",
    "django.contrib.auth.password_validation.NumericPasswordValidator",
)


DJANGO_EXCEPTION_REPORTERS: tuple[str, ...] = (
    "django.views.debug.ExceptionReporter",
)


DJANGO_BUILTIN_USER_MODELS: tuple[str, ...] = (
    "auth.User",
)


# Settings whose value is a list of strings.
_LIST_SETTINGS = frozenset({
    "INSTALLED_APPS", "MIDDLEWARE", "AUTHENTICATION_BACKENDS",
})

# Settings whose value is a single string.
_SCALAR_SETTINGS = frozenset({
    "DEFAULT_AUTO_FIELD", "AUTH_USER_MODEL", "WSGI_APPLICATION",
    "DEFAULT_EXCEPTION_REPORTER",
})

# List-of-dicts settings: we complete the value of a specific dict key.
_DICT_LIST_SETTINGS: dict[str, str] = {
    "AUTH_PASSWORD_VALIDATORS": "NAME",
}


_MARKER = "__iommi_lsp_settings_marker__"


class SettingsAnalyzer:
    """Implements the :class:`Analyzer` Protocol for Django-settings completion."""

    name = "settings"

    def __init__(
        self,
        workspace_root: Path,
        text_provider: Callable[[str], str | None] | None = None,
        django_index_provider: "Callable[[], DjangoIndex] | None" = None,
    ) -> None:
        self.workspace_root = workspace_root
        self._text_provider = text_provider
        self._django_index_provider = django_index_provider
        self._workspace_apps: list[str] = []
        self._wsgi_paths: list[str] = []

    @property
    def workspace_apps(self) -> list[str]:
        return list(self._workspace_apps)

    @property
    def wsgi_paths(self) -> list[str]:
        return list(self._wsgi_paths)

    # -- Analyzer protocol ----------------------------------------------------

    async def index(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self._workspace_apps = sorted(discover_workspace_apps(workspace_root))
        self._wsgi_paths = sorted(discover_wsgi_applications(workspace_root))
        _log.info(
            "indexed %d workspace apps, %d wsgi modules under %s",
            len(self._workspace_apps), len(self._wsgi_paths), workspace_root,
        )

    async def on_file_changed(self, uri: str) -> None:
        # One-shot discovery — adding/removing apps.py rarely happens
        # during an editing session; restart picks it up.
        return None

    def is_false_positive(self, uri: str, diagnostic: Diagnostic) -> bool:
        return False

    def additional_diagnostics(self, uri: str) -> list[Diagnostic]:
        return []

    def completions(self, uri: str, position: dict) -> CompletionResult:
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
            _log.exception("settings completion scanner crashed; emitting nothing")
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
        setting = _detect_setting_via_patch(source, ctx.start)
        if setting is None:
            return empty

        candidates = self._candidates_for(setting)
        if not candidates:
            # Recognised setting but no items to offer — still exclusive
            # so ty's free-form name completions don't backfill garbage.
            return CompletionResult(items=[], exclusive=True)

        items: list[dict] = []
        for value in candidates:
            if partial and not value.startswith(partial):
                continue
            items.append({
                "label": value,
                "kind": 21,   # CompletionItemKind.Constant
                "insertText": value,
                "detail": f"{setting}",
                "data": {"source": "iommi_lsp.settings", "setting": setting},
            })
        return CompletionResult(items=items, exclusive=True)

    def _candidates_for(self, setting: str) -> list[str]:
        if setting == "INSTALLED_APPS":
            return sorted(
                set(DJANGO_CONTRIB_APPS)
                | set(EXTRA_APPS)
                | set(self._workspace_apps)
            )
        if setting == "MIDDLEWARE":
            return list(DJANGO_MIDDLEWARE)
        if setting == "AUTHENTICATION_BACKENDS":
            return list(DJANGO_AUTH_BACKENDS)
        if setting == "DEFAULT_AUTO_FIELD":
            return list(DJANGO_AUTO_FIELDS)
        if setting == "AUTH_USER_MODEL":
            return sorted(
                set(DJANGO_BUILTIN_USER_MODELS) | set(self._workspace_user_models())
            )
        if setting == "WSGI_APPLICATION":
            return list(self._wsgi_paths)
        if setting == "AUTH_PASSWORD_VALIDATORS":
            return list(DJANGO_PASSWORD_VALIDATORS)
        if setting == "DEFAULT_EXCEPTION_REPORTER":
            return list(DJANGO_EXCEPTION_REPORTERS)
        return []

    def _workspace_user_models(self) -> list[str]:
        """Workspace models in ``app_label.ModelName`` form.

        Drawn from the Django index when available. The app label is the
        first segment of the model's module qualname — matching Django's
        convention that ``app/models.py`` lives at module ``app.models``
        and has ``app`` as its app label.
        """
        if self._django_index_provider is None:
            return []
        try:
            index = self._django_index_provider()
        except Exception:
            return []
        if index is None:
            return []
        out: list[str] = []
        for m in index.models.values():
            if getattr(m, "is_builtin", False):
                continue
            if getattr(m, "abstract", False):
                continue
            module = getattr(m, "module", "") or ""
            app_label = module.split(".", 1)[0] if module else ""
            if not app_label:
                continue
            out.append(f"{app_label}.{m.name}")
        return out

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
# Workspace discovery.
# ---------------------------------------------------------------------------


def discover_workspace_apps(workspace_root: Path) -> set[str]:
    """Return every workspace package that looks like a Django app.

    The signal: a directory containing ``apps.py``. The result is the
    package's dotted import path computed relative to *workspace_root*.
    When ``apps.py`` declares an ``AppConfig`` subclass with a ``name``
    attribute, that name wins (it's the canonical, ``INSTALLED_APPS``-
    ready value); otherwise we fall back to the directory's relative
    dotted path. Both forms are returned when they differ — completion
    is non-destructive, the user picks.
    """
    out: set[str] = set()
    root = workspace_root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        if "apps.py" not in filenames:
            continue
        app_dir = Path(dirpath)
        # The directory components form a valid dotted path only if
        # every segment is a Python identifier.
        try:
            rel = app_dir.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if parts and all(p.isidentifier() for p in parts):
            out.add(".".join(parts))
        # Read apps.py for the AppConfig.name override.
        declared = _read_appconfig_name(app_dir / "apps.py")
        if declared:
            out.add(declared)
    return out


def discover_wsgi_applications(workspace_root: Path) -> set[str]:
    """Find ``wsgi.py`` files that define ``application = …`` and return
    their dotted ``WSGI_APPLICATION`` path (``pkg.wsgi.application``).
    """
    out: set[str] = set()
    root = workspace_root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        if "wsgi.py" not in filenames:
            continue
        wsgi_path = Path(dirpath) / "wsgi.py"
        try:
            source = wsgi_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        if not _module_defines_name(tree, "application"):
            continue
        try:
            rel = Path(dirpath).relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if parts and all(p.isidentifier() for p in parts):
            out.add(".".join(parts) + ".wsgi.application")
    return out


def _read_appconfig_name(apps_py: Path) -> str | None:
    try:
        source = apps_py.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        # Cheap check: the class's bases include something ending in
        # ``AppConfig``. We don't follow imports — close enough.
        if not _bases_look_like_appconfig(node):
            continue
        for stmt in node.body:
            target_id = _simple_assign_target(stmt)
            if target_id != "name":
                continue
            value = _string_value(stmt)
            if value:
                return value
    return None


def _bases_look_like_appconfig(node: ast.ClassDef) -> bool:
    for b in node.bases:
        if isinstance(b, ast.Name) and b.id == "AppConfig":
            return True
        if isinstance(b, ast.Attribute) and b.attr == "AppConfig":
            return True
    return False


def _simple_assign_target(stmt: ast.AST) -> str | None:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        t = stmt.targets[0]
        if isinstance(t, ast.Name):
            return t.id
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id
    return None


def _string_value(stmt: ast.AST) -> str | None:
    value = getattr(stmt, "value", None)
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _module_defines_name(tree: ast.Module, name: str) -> bool:
    for stmt in tree.body:
        if _simple_assign_target(stmt) == name:
            return True
    return False


# ---------------------------------------------------------------------------
# AST-based setting-context detection.
# ---------------------------------------------------------------------------


def _detect_setting_via_patch(source: str, string_start: int) -> str | None:
    """Identify which Django setting the open string at *string_start* belongs to.

    Replaces the open string (from its opening quote onward) with a
    valid string-literal sentinel, balances any open brackets, and
    parses the result. Walks the parse tree to find the sentinel and
    figure out the enclosing assignment target.

    When the immediate enclosing unmatched bracket in *head* is ``(``,
    we append a trailing comma so that ``SETTING = ('`` becomes a real
    one-element tuple rather than a grouped scalar — otherwise Python's
    parser collapses the parens away and we can't tell the user meant
    a tuple-form list.

    Returns the setting name (one of the recognised set) or ``None`` if
    the position doesn't match any.
    """
    head = source[:string_start]
    suffix = "," if _enclosing_bracket_is_paren(head) else ""
    inserted = f'"{_MARKER}"{suffix}'
    closes = _close_brackets(head + inserted)
    patched = head + inserted + closes
    try:
        tree = ast.parse(patched)
    except SyntaxError:
        return None

    path = _find_marker_path(tree)
    if path is None:
        return None
    return _classify_setting_from_path(path)


def _enclosing_bracket_is_paren(src: str) -> bool:
    """Return True if the deepest unmatched open bracket in *src* is ``(``."""
    stack: list[str] = []
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
            stack.append(ch)
        elif ch in ")]}":
            if stack:
                stack.pop()
        elif ch == "#":
            j = src.find("\n", i)
            i = n if j == -1 else j
            continue
        i += 1
    return bool(stack) and stack[-1] == "("


def _find_marker_path(tree: ast.Module) -> list[ast.AST] | None:
    found: list[list[ast.AST]] = []

    def visit(node: ast.AST, trail: list[ast.AST]) -> None:
        if isinstance(node, ast.Constant) and node.value == _MARKER:
            found.append(trail + [node])
            return
        for child in ast.iter_child_nodes(node):
            visit(child, trail + [node])

    visit(tree, [])
    return found[0] if len(found) == 1 else (found[0] if found else None)


def _classify_setting_from_path(path: list[ast.AST]) -> str | None:
    """Walk *path* upward from the sentinel and return the setting name."""
    # Path is root...marker. We look for the nearest enclosing
    # Assign/AugAssign whose target is a known setting name, and verify
    # the shape of the path between the assignment and the marker.
    for i in range(len(path) - 2, -1, -1):
        node = path[i]
        setting = _setting_name_from_assignment(node)
        if setting is None:
            continue
        between = path[i + 1: -1]   # nodes strictly between assignment and the marker
        # Skip the assignment's own value wrapper if present
        # (e.g. ``Assign.value`` link doesn't appear in the path — children
        # are direct subtrees so the value's root *is* path[i+1]).
        if setting in _LIST_SETTINGS:
            if _path_is_list_element(between):
                return setting
            return None
        if setting in _SCALAR_SETTINGS:
            if not between:
                return setting
            return None
        key = _DICT_LIST_SETTINGS.get(setting)
        if key is not None:
            if _path_is_dict_value_in_list(between, key):
                return setting
            return None
    return None


def _setting_name_from_assignment(node: ast.AST) -> str | None:
    """Return the setting name if *node* is ``SETTING = …`` or ``SETTING += …``."""
    candidates = _LIST_SETTINGS | _SCALAR_SETTINGS | set(_DICT_LIST_SETTINGS)
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in candidates:
                return t.id
    if isinstance(node, ast.AugAssign):
        if isinstance(node.target, ast.Name) and node.target.id in candidates:
            return node.target.id
    return None


def _path_is_list_element(between: list[ast.AST]) -> bool:
    """The marker is a direct element of a list/tuple within the
    assignment's value subtree.

    Allows transparent wrappers like ``BinOp`` so patterns such as
    ``INSTALLED_APPS = ["myapp"] + extras`` still trigger.
    """
    if not between:
        return False
    return isinstance(between[-1], (ast.List, ast.Tuple))


def _path_is_dict_value_in_list(between: list[ast.AST], key: str) -> bool:
    """Match ``[..., {key: marker, ...}, ...]`` — used by AUTH_PASSWORD_VALIDATORS."""
    if len(between) < 2:
        return False
    inner = between[-1]
    if not isinstance(inner, ast.Dict):
        return False
    if not any(isinstance(b, (ast.List, ast.Tuple)) for b in between[:-1]):
        return False
    for k, v in zip(inner.keys, inner.values):
        if (
            isinstance(k, ast.Constant)
            and k.value == key
            and isinstance(v, ast.Constant)
            and v.value == _MARKER
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Source-utility helpers (mirrors of the ones in the templates analyzer).
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


class _StringCtx:
    __slots__ = ("quote", "start")

    def __init__(self, quote: str, start: int) -> None:
        self.quote = quote
        self.start = start


def _string_state_at(source: str, offset: int) -> _StringCtx | None:
    """Return the open single-line string at *offset*, or ``None``."""
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
                return None
            i += 1
            continue
        if ch in '"\'':
            if (
                i + 2 < n
                and source[i + 1] == ch
                and source[i + 2] == ch
            ):
                # Triple-quoted span. Find the matching close and skip
                # past it — settings.py modules commonly open with a
                # docstring, and bailing here would suppress completion
                # for the rest of the file.
                closing = source.find(ch * 3, i + 3, n)
                if closing == -1:
                    # Unterminated triple-quote (or it closes past the
                    # cursor) — the cursor is inside a multi-line string,
                    # not a single-line one.
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


def _close_brackets(src: str) -> str:
    """Return the closing tokens needed to balance *src*."""
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
