"""Tests for SettingsAnalyzer — Django-settings string-value completion."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from iommi_lsp.analyzers.settings import SettingsAnalyzer, discover_workspace_apps
from iommi_lsp.analyzers.settings.analyzer import (
    discover_wsgi_applications,
)


def _write_with_cursor(
    tmp_path: Path, src_before: str, src_after: str = "",
    filename: str = "settings.py",
) -> tuple[str, dict]:
    f = tmp_path / filename
    f.write_text(src_before + src_after)
    line = src_before.count("\n")
    last_nl = src_before.rfind("\n")
    character = len(src_before) - (last_nl + 1)
    return f.as_uri(), {"line": line, "character": character}


def _labels(result) -> list[str]:
    return [it["label"] for it in result.items]


@pytest.fixture
def analyzer(tmp_path: Path) -> SettingsAnalyzer:
    a = SettingsAnalyzer(workspace_root=tmp_path)
    asyncio.run(a.index(tmp_path))
    return a


# ---------------------------------------------------------------------------
# discover_workspace_apps
# ---------------------------------------------------------------------------


def test_discover_apps_via_apps_py(tmp_path: Path) -> None:
    (tmp_path / "myapp").mkdir()
    (tmp_path / "myapp" / "apps.py").write_text("")
    assert discover_workspace_apps(tmp_path) >= {"myapp"}


def test_discover_uses_appconfig_name(tmp_path: Path) -> None:
    (tmp_path / "myapp").mkdir()
    (tmp_path / "myapp" / "apps.py").write_text(
        "from django.apps import AppConfig\n"
        "class MyAppConfig(AppConfig):\n"
        "    name = 'project.myapp'\n"
    )
    found = discover_workspace_apps(tmp_path)
    # Both the relative dotted path and the declared name appear.
    assert "myapp" in found
    assert "project.myapp" in found


def test_discover_skips_venv(tmp_path: Path) -> None:
    (tmp_path / ".venv" / "lib" / "fakeapp").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "fakeapp" / "apps.py").write_text("")
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "apps.py").write_text("")
    assert discover_workspace_apps(tmp_path) == {"real"}


def test_discover_wsgi_applications(tmp_path: Path) -> None:
    (tmp_path / "project").mkdir()
    (tmp_path / "project" / "wsgi.py").write_text(
        "application = lambda *a, **kw: None\n"
    )
    assert discover_wsgi_applications(tmp_path) == {"project.wsgi.application"}


def test_discover_wsgi_skips_files_without_application(tmp_path: Path) -> None:
    (tmp_path / "project").mkdir()
    (tmp_path / "project" / "wsgi.py").write_text("# empty\n")
    assert discover_wsgi_applications(tmp_path) == set()


# ---------------------------------------------------------------------------
# INSTALLED_APPS
# ---------------------------------------------------------------------------


def test_installed_apps_offers_django_contrib(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "INSTALLED_APPS = ['")
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    labels = set(_labels(result))
    assert "django.contrib.admin" in labels
    assert "django.contrib.auth" in labels
    # iommi too — it's a hard dep of iommi_lsp.
    assert "iommi" in labels


def test_installed_apps_filters_by_prefix(analyzer, tmp_path: Path) -> None:
    # Editor-side filtering proved unreliable (many LSP clients treat
    # ``.`` as a word boundary), so we filter server-side: completion
    # items must all be prefixed by the user's in-string text.
    uri, pos = _write_with_cursor(tmp_path, "INSTALLED_APPS = ['django.contrib.au")
    labels = _labels(analyzer.completions(uri, pos))
    assert "django.contrib.auth" in labels
    assert "django.contrib.admin" not in labels   # ``ad`` ≠ prefix of ``au``


def test_installed_apps_filters_by_prefix_workspace_app(tmp_path: Path) -> None:
    # Regression: typing ``'dryft.ba`` inside INSTALLED_APPS must NOT
    # return every candidate — only items whose value prefix-matches
    # ``dryft.ba`` (i.e. ``dryft.base`` and any nested ``dryft.ba*``).
    # We've broken this twice now by trusting the editor to filter
    # client-side; this test pins the contract at the analyzer boundary.
    (tmp_path / "dryft" / "base").mkdir(parents=True)
    (tmp_path / "dryft" / "base" / "apps.py").write_text(
        "from django.apps import AppConfig\n"
        "class BaseConfig(AppConfig):\n"
        "    name = 'dryft.base'\n"
    )
    (tmp_path / "dryft" / "core").mkdir(parents=True)
    (tmp_path / "dryft" / "core" / "apps.py").write_text("")
    a = SettingsAnalyzer(workspace_root=tmp_path)
    asyncio.run(a.index(tmp_path))

    uri, pos = _write_with_cursor(tmp_path, "INSTALLED_APPS = ['dryft.ba")
    labels = _labels(a.completions(uri, pos))
    # The match — workspace AppConfig declares this name.
    assert "dryft.base" in labels
    # Built-in django.contrib.* candidates must NOT slip through.
    assert "django.contrib.admin" not in labels
    assert "django.contrib.auth" not in labels
    # Sibling workspace app that doesn't share the prefix must NOT
    # appear either.
    assert "dryft.core" not in labels
    # And the iommi extra app — also no.
    assert "iommi" not in labels
    # Every returned label is a real prefix match.
    for label in labels:
        assert label.startswith("dryft.ba"), label


def test_installed_apps_offers_workspace_app(tmp_path: Path) -> None:
    (tmp_path / "myapp").mkdir()
    (tmp_path / "myapp" / "apps.py").write_text("")
    a = SettingsAnalyzer(workspace_root=tmp_path)
    asyncio.run(a.index(tmp_path))
    uri, pos = _write_with_cursor(tmp_path, "INSTALLED_APPS = ['")
    labels = set(_labels(a.completions(uri, pos)))
    assert "myapp" in labels


def test_installed_apps_in_tuple(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "INSTALLED_APPS = ('")
    labels = set(_labels(analyzer.completions(uri, pos)))
    assert "django.contrib.admin" in labels


def test_installed_apps_with_aug_assign(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "INSTALLED_APPS += ['")
    labels = set(_labels(analyzer.completions(uri, pos)))
    assert "django.contrib.admin" in labels


def test_installed_apps_with_existing_entries(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(
        tmp_path,
        "INSTALLED_APPS = [\n    'django.contrib.admin',\n    '",
        "\n]\n",
    )
    labels = set(_labels(analyzer.completions(uri, pos)))
    assert "django.contrib.auth" in labels


def test_installed_apps_with_binop_concat(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "INSTALLED_APPS = ['")
    labels = set(_labels(analyzer.completions(
        uri, pos,
    )))
    assert "django.contrib.admin" in labels


def test_installed_apps_after_module_docstring(analyzer, tmp_path: Path) -> None:
    """Triple-quoted module docstrings above INSTALLED_APPS must not
    suppress completion. Django's startproject-generated settings.py
    opens with one, so this is the common case."""
    uri, pos = _write_with_cursor(
        tmp_path,
        '"""\nDjango settings for blah.\n"""\n\nINSTALLED_APPS = [\n    \'',
    )
    labels = set(_labels(analyzer.completions(uri, pos)))
    assert "django.contrib.admin" in labels


def test_installed_apps_nested_workspace_app_with_long_prefix(
    tmp_path: Path,
) -> None:
    """A long realistic INSTALLED_APPS followed by ``'dryft.b`` should
    surface the nested-package workspace app ``dryft.base``."""
    (tmp_path / "dryft" / "base").mkdir(parents=True)
    (tmp_path / "dryft" / "base" / "apps.py").write_text(
        "from django.apps import AppConfig\n"
        "class BaseConfig(AppConfig):\n"
        "    name = 'dryft.base'\n"
    )
    a = SettingsAnalyzer(workspace_root=tmp_path)
    asyncio.run(a.index(tmp_path))

    before = (
        "INSTALLED_APPS = [\n"
        "    'servestatic.runserver_nostatic',\n"
        "    'django_fastdev',\n"
        "    'django.contrib.auth',\n"
        "    'django.contrib.contenttypes',\n"
        "    'django.contrib.sessions',\n"
        "    'django.contrib.messages',\n"
        "    'django.contrib.staticfiles',\n"
        "    'django.contrib.sites',\n"
        "    'django.contrib.postgres',\n"
        "    'allauth',\n"
        "    'iommi',\n"
        "    'dryft.b"
    )
    uri, pos = _write_with_cursor(tmp_path, before)
    result = a.completions(uri, pos)
    labels = set(_labels(result))
    assert "dryft.base" in labels, labels


# ---------------------------------------------------------------------------
# MIDDLEWARE
# ---------------------------------------------------------------------------


def test_middleware_offers_django_middleware(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "MIDDLEWARE = ['")
    labels = set(_labels(analyzer.completions(uri, pos)))
    assert "django.middleware.csrf.CsrfViewMiddleware" in labels
    assert "django.middleware.security.SecurityMiddleware" in labels


def test_middleware_filters_by_prefix(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(
        tmp_path, "MIDDLEWARE = ['django.middleware.csrf"
    )
    labels = _labels(analyzer.completions(uri, pos))
    # The only middleware whose dotted name prefixes ``django.middleware.csrf``.
    assert labels == ["django.middleware.csrf.CsrfViewMiddleware"]


# ---------------------------------------------------------------------------
# AUTHENTICATION_BACKENDS
# ---------------------------------------------------------------------------


def test_auth_backends(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "AUTHENTICATION_BACKENDS = ['")
    labels = set(_labels(analyzer.completions(uri, pos)))
    assert "django.contrib.auth.backends.ModelBackend" in labels


# ---------------------------------------------------------------------------
# DEFAULT_AUTO_FIELD
# ---------------------------------------------------------------------------


def test_default_auto_field(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "DEFAULT_AUTO_FIELD = '")
    labels = set(_labels(analyzer.completions(uri, pos)))
    assert "django.db.models.BigAutoField" in labels
    assert "django.db.models.AutoField" in labels


def test_default_auto_field_not_in_list_context(analyzer, tmp_path: Path) -> None:
    # ``DEFAULT_AUTO_FIELD = ['x']`` is malformed — our scalar setting
    # check requires the marker to be the *direct* value of the assignment.
    # No setting matches, so we let ty handle this position.
    uri, pos = _write_with_cursor(tmp_path, "DEFAULT_AUTO_FIELD = ['")
    result = analyzer.completions(uri, pos)
    assert result.items == []
    assert result.exclusive is False


# ---------------------------------------------------------------------------
# AUTH_USER_MODEL
# ---------------------------------------------------------------------------


def _fake_index(*models: dict) -> Any:
    class _FakeIndex:
        def __init__(self) -> None:
            self.models = {
                m["qualname"]: SimpleNamespace(
                    qualname=m["qualname"],
                    module=m["module"],
                    name=m["name"],
                    is_builtin=m.get("is_builtin", False),
                    abstract=m.get("abstract", False),
                )
                for m in models
            }
    return _FakeIndex()


def test_auth_user_model_offers_builtin(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "AUTH_USER_MODEL = '")
    labels = set(_labels(analyzer.completions(uri, pos)))
    assert "auth.User" in labels


def test_auth_user_model_offers_workspace_models(tmp_path: Path) -> None:
    index = _fake_index(
        {"qualname": "myapp.models.User", "module": "myapp.models", "name": "User"},
        {"qualname": "blog.models.Author", "module": "blog.models", "name": "Author"},
    )
    a = SettingsAnalyzer(
        workspace_root=tmp_path,
        django_index_provider=lambda: index,
    )
    asyncio.run(a.index(tmp_path))
    uri, pos = _write_with_cursor(tmp_path, "AUTH_USER_MODEL = '")
    labels = set(_labels(a.completions(uri, pos)))
    assert "myapp.User" in labels
    assert "blog.Author" in labels


def test_auth_user_model_skips_abstract_and_builtin(tmp_path: Path) -> None:
    index = _fake_index(
        {
            "qualname": "django.contrib.auth.models.User",
            "module": "django.contrib.auth.models",
            "name": "User",
            "is_builtin": True,
        },
        {
            "qualname": "core.models.Base",
            "module": "core.models",
            "name": "Base",
            "abstract": True,
        },
        {
            "qualname": "core.models.Member",
            "module": "core.models",
            "name": "Member",
        },
    )
    a = SettingsAnalyzer(
        workspace_root=tmp_path,
        django_index_provider=lambda: index,
    )
    asyncio.run(a.index(tmp_path))
    uri, pos = _write_with_cursor(tmp_path, "AUTH_USER_MODEL = '")
    labels = set(_labels(a.completions(uri, pos)))
    assert "core.Member" in labels
    assert "core.Base" not in labels
    # Builtin still surfaces via the static "auth.User" list, but the
    # qualname-derived `django.User` shouldn't sneak in from the index.
    assert "django.User" not in labels


# ---------------------------------------------------------------------------
# WSGI_APPLICATION
# ---------------------------------------------------------------------------


def test_wsgi_application(tmp_path: Path) -> None:
    (tmp_path / "project").mkdir()
    (tmp_path / "project" / "wsgi.py").write_text(
        "application = lambda *a, **kw: None\n"
    )
    a = SettingsAnalyzer(workspace_root=tmp_path)
    asyncio.run(a.index(tmp_path))
    uri, pos = _write_with_cursor(tmp_path, "WSGI_APPLICATION = '")
    labels = set(_labels(a.completions(uri, pos)))
    assert "project.wsgi.application" in labels


# ---------------------------------------------------------------------------
# AUTH_PASSWORD_VALIDATORS
# ---------------------------------------------------------------------------


def test_auth_password_validators(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(
        tmp_path,
        "AUTH_PASSWORD_VALIDATORS = [\n    {'NAME': '",
        "'},\n]\n",
    )
    labels = set(_labels(analyzer.completions(uri, pos)))
    assert (
        "django.contrib.auth.password_validation.MinimumLengthValidator"
        in labels
    )


def test_auth_password_validators_wrong_key(analyzer, tmp_path: Path) -> None:
    # Only the ``NAME`` value triggers — random other keys don't.
    uri, pos = _write_with_cursor(
        tmp_path,
        "AUTH_PASSWORD_VALIDATORS = [\n    {'OPTIONS': '",
    )
    result = analyzer.completions(uri, pos)
    assert result.items == []
    assert result.exclusive is False


# ---------------------------------------------------------------------------
# DEFAULT_EXCEPTION_REPORTER
# ---------------------------------------------------------------------------


def test_default_exception_reporter(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "DEFAULT_EXCEPTION_REPORTER = '")
    labels = set(_labels(analyzer.completions(uri, pos)))
    assert "django.views.debug.ExceptionReporter" in labels


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_unrelated_assignment_does_not_trigger(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "DEBUG_TOOLS = ['")
    result = analyzer.completions(uri, pos)
    assert result.items == []
    assert result.exclusive is False


def test_outside_string_does_not_trigger(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "INSTALLED_APPS = [")
    result = analyzer.completions(uri, pos)
    assert result.items == []
    assert result.exclusive is False


def test_uses_text_provider_over_disk(tmp_path: Path) -> None:
    docs: dict[str, str] = {}
    a = SettingsAnalyzer(
        workspace_root=tmp_path,
        text_provider=lambda uri: docs.get(uri),
    )
    asyncio.run(a.index(tmp_path))
    f = tmp_path / "settings.py"
    f.write_text("# empty\n")
    uri = f.as_uri()
    docs[uri] = "INSTALLED_APPS = ['"
    pos = {"line": 0, "character": len("INSTALLED_APPS = ['")}
    labels = set(_labels(a.completions(uri, pos)))
    assert "django.contrib.admin" in labels
