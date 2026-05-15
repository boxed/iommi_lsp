"""Tests for UrlAnalyzer — URL-name completion + diagnostics."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from iommi_lsp.analyzers.urls import (
    UrlAnalyzer,
    build_url_index,
    discover_urls,
)
from iommi_lsp.analyzers.urls.analyzer import (
    URL_DIAG_CODE,
    _scan_diagnostics,
)


def _write_with_cursor(
    tmp_path: Path, src_before: str, src_after: str = "",
    filename: str = "u.py",
) -> tuple[str, dict]:
    f = tmp_path / filename
    f.write_text(src_before + src_after)
    line = src_before.count("\n")
    last_nl = src_before.rfind("\n")
    character = len(src_before) - (last_nl + 1)
    return f.as_uri(), {"line": line, "character": character}


def _labels(result) -> list[str]:
    return [it["label"] for it in result.items]


# ---------------------------------------------------------------------------
# discover_urls / build_url_index
# ---------------------------------------------------------------------------


def test_discover_urls_finds_every_urls_py(tmp_path: Path) -> None:
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "urls.py").write_text("urlpatterns = []\n")
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "urls.py").write_text("urlpatterns = []\n")
    found = {p.name for p in discover_urls(tmp_path)}
    assert found == {"urls.py"}
    assert len(discover_urls(tmp_path)) == 2


def test_discover_urls_skips_venv(tmp_path: Path) -> None:
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "urls.py").write_text("urlpatterns = []\n")
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "urls.py").write_text("urlpatterns = []\n")
    found = discover_urls(tmp_path)
    assert len(found) == 1
    assert found[0].parent.name == "real"


def test_build_url_index_collects_path_names(tmp_path: Path) -> None:
    (tmp_path / "myapp").mkdir()
    (tmp_path / "myapp" / "urls.py").write_text(
        "from django.urls import path\n"
        "urlpatterns = [\n"
        "    path('', view, name='index'),\n"
        "    path('about/', view, name='about'),\n"
        "]\n"
    )
    index = build_url_index(tmp_path)
    assert index.names == {"index", "about"}


def test_build_url_index_re_path_and_url(tmp_path: Path) -> None:
    (tmp_path / "urls.py").write_text(
        "from django.urls import re_path\n"
        "from django.conf.urls import url\n"
        "urlpatterns = [\n"
        "    re_path(r'^a/$', view, name='alpha'),\n"
        "    url(r'^b/$', view, name='beta'),\n"
        "]\n"
    )
    index = build_url_index(tmp_path)
    assert index.names == {"alpha", "beta"}


def test_build_url_index_handles_no_name(tmp_path: Path) -> None:
    (tmp_path / "urls.py").write_text(
        "from django.urls import path\n"
        "urlpatterns = [path('foo/', view)]\n"
    )
    assert build_url_index(tmp_path).names == set()


def test_build_url_index_app_name_namespacing(tmp_path: Path) -> None:
    # Root urls.py includes blog.urls; blog/urls.py declares app_name = 'blog'.
    (tmp_path / "urls.py").write_text(
        "from django.urls import path, include\n"
        "urlpatterns = [path('blog/', include('blog.urls'))]\n"
    )
    (tmp_path / "blog").mkdir()
    (tmp_path / "blog" / "__init__.py").write_text("")
    (tmp_path / "blog" / "urls.py").write_text(
        "from django.urls import path\n"
        "app_name = 'blog'\n"
        "urlpatterns = [path('', view, name='index')]\n"
    )
    index = build_url_index(tmp_path)
    # The bare 'index' is registered too (we don't try to suppress it —
    # bias toward leniency in completion).
    assert "blog:index" in index.names
    assert "index" in index.names


def test_build_url_index_explicit_namespace_override(tmp_path: Path) -> None:
    (tmp_path / "urls.py").write_text(
        "from django.urls import path, include\n"
        "urlpatterns = [\n"
        "    path('b/', include('blog.urls', namespace='posts')),\n"
        "]\n"
    )
    (tmp_path / "blog").mkdir()
    (tmp_path / "blog" / "__init__.py").write_text("")
    (tmp_path / "blog" / "urls.py").write_text(
        "from django.urls import path\n"
        "app_name = 'blog'\n"   # overridden by namespace=
        "urlpatterns = [path('', view, name='detail')]\n"
    )
    index = build_url_index(tmp_path)
    assert "posts:detail" in index.names


# ---------------------------------------------------------------------------
# UrlAnalyzer.completions
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer(tmp_path: Path) -> UrlAnalyzer:
    (tmp_path / "myapp").mkdir()
    (tmp_path / "myapp" / "urls.py").write_text(
        "from django.urls import path\n"
        "urlpatterns = [\n"
        "    path('', view, name='index'),\n"
        "    path('about/', view, name='about'),\n"
        "    path('contact/', view, name='contact'),\n"
        "]\n"
    )
    a = UrlAnalyzer(workspace_root=tmp_path)
    asyncio.run(a.index(tmp_path))
    return a


def test_completion_inside_reverse(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(
        tmp_path, "from django.urls import reverse\nreverse('"
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    assert set(_labels(result)) == {"about", "contact", "index"}


def test_completion_inside_redirect(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(
        tmp_path, "from django.shortcuts import redirect\nredirect('"
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    assert "about" in _labels(result)


def test_completion_inside_resolve_url(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(
        tmp_path,
        "from django.shortcuts import resolve_url\nresolve_url('",
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    assert "index" in _labels(result)


def test_completion_partial_filters(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "reverse('ab")
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    assert _labels(result) == ["about"]


def test_completion_outside_reverse_silent(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "x = '")
    result = analyzer.completions(uri, pos)
    # Not at a reverse-call site — let other analyzers / ty handle it.
    assert result.items == []
    assert result.exclusive is False


def test_completion_inside_reverse_lazy(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "reverse_lazy('")
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    assert "index" in _labels(result)


# ---------------------------------------------------------------------------
# UrlAnalyzer.additional_diagnostics
# ---------------------------------------------------------------------------


def test_diagnostic_unknown_reverse(analyzer, tmp_path: Path) -> None:
    src = "from django.urls import reverse\nreverse('nope')\n"
    f = tmp_path / "use.py"
    f.write_text(src)
    diags = analyzer.additional_diagnostics(f.as_uri())
    assert len(diags) == 1
    d = diags[0]
    assert d["code"] == URL_DIAG_CODE
    assert "nope" in d["message"]


def test_diagnostic_known_name_silent(analyzer, tmp_path: Path) -> None:
    src = "from django.urls import reverse\nreverse('about')\n"
    f = tmp_path / "use.py"
    f.write_text(src)
    assert analyzer.additional_diagnostics(f.as_uri()) == []


def test_diagnostic_redirect_to_path_is_silent(analyzer, tmp_path: Path) -> None:
    # ``redirect('/some/path/')`` is a URL path, not a name.
    src = "from django.shortcuts import redirect\nredirect('/foo/bar/')\n"
    f = tmp_path / "use.py"
    f.write_text(src)
    assert analyzer.additional_diagnostics(f.as_uri()) == []


def test_diagnostic_redirect_to_full_url_is_silent(
    analyzer, tmp_path: Path,
) -> None:
    src = "from django.shortcuts import redirect\nredirect('https://x.com')\n"
    f = tmp_path / "use.py"
    f.write_text(src)
    assert analyzer.additional_diagnostics(f.as_uri()) == []


def test_diagnostic_redirect_to_dot_is_silent(analyzer, tmp_path: Path) -> None:
    src = "from django.shortcuts import redirect\nredirect('.')\n"
    f = tmp_path / "use.py"
    f.write_text(src)
    assert analyzer.additional_diagnostics(f.as_uri()) == []


def test_diagnostic_redirect_to_dotdot_is_silent(analyzer, tmp_path: Path) -> None:
    src = "from django.shortcuts import redirect\nredirect('..')\n"
    f = tmp_path / "use.py"
    f.write_text(src)
    assert analyzer.additional_diagnostics(f.as_uri()) == []


def test_diagnostic_variable_arg_is_silent(analyzer, tmp_path: Path) -> None:
    src = "from django.urls import reverse\nx = 'about'\nreverse(x)\n"
    f = tmp_path / "use.py"
    f.write_text(src)
    assert analyzer.additional_diagnostics(f.as_uri()) == []


def test_diagnostic_close_match_suggestion(analyzer, tmp_path: Path) -> None:
    src = "from django.urls import reverse\nreverse('abou')\n"
    f = tmp_path / "use.py"
    f.write_text(src)
    diags = analyzer.additional_diagnostics(f.as_uri())
    assert diags
    assert "about" in diags[0]["message"]


def test_diagnostic_namespaced_unknown(tmp_path: Path) -> None:
    (tmp_path / "blog").mkdir()
    (tmp_path / "blog" / "__init__.py").write_text("")
    (tmp_path / "blog" / "urls.py").write_text(
        "from django.urls import path\n"
        "app_name = 'blog'\n"
        "urlpatterns = [path('', view, name='detail')]\n"
    )
    (tmp_path / "urls.py").write_text(
        "from django.urls import path, include\n"
        "urlpatterns = [path('b/', include('blog.urls'))]\n"
    )
    a = UrlAnalyzer(workspace_root=tmp_path)
    asyncio.run(a.index(tmp_path))
    src = "from django.urls import reverse\nreverse('blog:nope')\n"
    f = tmp_path / "use.py"
    f.write_text(src)
    diags = a.additional_diagnostics(f.as_uri())
    assert diags and "blog:nope" in diags[0]["message"]
    # Known namespaced name is silent.
    src2 = "reverse('blog:detail')\n"
    f2 = tmp_path / "use2.py"
    f2.write_text(src2)
    assert a.additional_diagnostics(f2.as_uri()) == []


def test_empty_index_no_diagnostics(tmp_path: Path) -> None:
    a = UrlAnalyzer(workspace_root=tmp_path)
    asyncio.run(a.index(tmp_path))
    src = "reverse('whatever')\n"
    f = tmp_path / "use.py"
    f.write_text(src)
    # No urls.py in workspace -> empty index -> stay quiet.
    assert a.additional_diagnostics(f.as_uri()) == []


# ---------------------------------------------------------------------------
# on_file_changed
# ---------------------------------------------------------------------------


def test_on_file_changed_rescans_urls_py(tmp_path: Path) -> None:
    urls = tmp_path / "app" / "urls.py"
    urls.parent.mkdir()
    urls.write_text(
        "from django.urls import path\n"
        "urlpatterns = [path('', view, name='alpha')]\n"
    )
    a = UrlAnalyzer(workspace_root=tmp_path)
    asyncio.run(a.index(tmp_path))
    assert "alpha" in a.names

    urls.write_text(
        "from django.urls import path\n"
        "urlpatterns = [\n"
        "    path('', view, name='alpha'),\n"
        "    path('b/', view, name='beta'),\n"
        "]\n"
    )
    asyncio.run(a.on_file_changed(urls.as_uri()))
    assert "alpha" in a.names
    assert "beta" in a.names


def test_on_file_changed_ignores_non_urls_files(tmp_path: Path) -> None:
    (tmp_path / "urls.py").write_text(
        "from django.urls import path\n"
        "urlpatterns = [path('', view, name='only')]\n"
    )
    a = UrlAnalyzer(workspace_root=tmp_path)
    asyncio.run(a.index(tmp_path))
    other = tmp_path / "views.py"
    other.write_text("")
    asyncio.run(a.on_file_changed(other.as_uri()))
    assert "only" in a.names
