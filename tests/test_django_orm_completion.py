"""Tests for DjangoAnalyzer.completions — ORM-kwarg completion items."""

from __future__ import annotations

from pathlib import Path

import pytest

from iommi_lsp.analyzers.django import DjangoAnalyzer, build_index


CORPUS = Path(__file__).parent / "corpus"


@pytest.fixture
def analyzer_basic() -> DjangoAnalyzer:
    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")
    return a


@pytest.fixture
def analyzer_blog() -> DjangoAnalyzer:
    a = DjangoAnalyzer(workspace_root=CORPUS / "related_names")
    a.django_index = build_index(CORPUS / "related_names")
    return a


def _write_with_cursor(tmp_path: Path, src_before_cursor: str, src_after_cursor: str = "") -> tuple[str, dict]:
    """Write a file and return (uri, position) where position points at the cursor.

    Cursor's line/character is computed from ``src_before_cursor``: the line is
    the number of newlines before the cursor, and the character is the column
    of the cursor within that line.
    """
    f = tmp_path / "u.py"
    f.write_text(src_before_cursor + src_after_cursor)
    line = src_before_cursor.count("\n")
    last_nl = src_before_cursor.rfind("\n")
    character = len(src_before_cursor) - (last_nl + 1)
    return f.as_uri(), {"line": line, "character": character}


def _labels(result) -> list[str]:
    return [it["label"] for it in result.items]


def test_completion_empty_partial_returns_all_fields(analyzer_basic, tmp_path):
    uri, pos = _write_with_cursor(
        tmp_path, "from myapp.models import User\nUser.objects.filter("
    )
    result = analyzer_basic.completions(uri, pos)
    assert result.exclusive is True
    labels = set(_labels(result))
    assert "username" in labels
    assert "email" in labels
    assert "pk" in labels
    # `profile_set` reverse from Profile -> User.
    assert "profile_set" in labels


def test_completion_partial_prefix_filters(analyzer_basic, tmp_path):
    uri, pos = _write_with_cursor(
        tmp_path, "from myapp.models import User\nUser.objects.filter(em"
    )
    result = analyzer_basic.completions(uri, pos)
    assert result.exclusive is True
    assert _labels(result) == ["email"]
    assert result.items[0]["insertText"] == "email="
    assert result.items[0]["kind"] == 5   # CompletionItemKind.Field
    assert result.items[0]["detail"] == "EmailField"


def test_completion_after_existing_kwarg(analyzer_basic, tmp_path):
    uri, pos = _write_with_cursor(
        tmp_path,
        "from myapp.models import User\nUser.objects.filter(username='a', em",
    )
    result = analyzer_basic.completions(uri, pos)
    assert result.exclusive is True
    assert _labels(result) == ["email"]


def test_completion_inside_chained_call(analyzer_basic, tmp_path):
    # Chained filter().exclude() — cursor is in exclude()'s kwargs.
    uri, pos = _write_with_cursor(
        tmp_path,
        "from myapp.models import User\n"
        "User.objects.filter(username='a').exclude(",
    )
    result = analyzer_basic.completions(uri, pos)
    assert result.exclusive is True
    assert "email" in _labels(result)


def test_completion_outside_lookup_method_silent(analyzer_basic, tmp_path):
    # `annotate` isn't in our lookup-method set; we shouldn't fire.
    uri, pos = _write_with_cursor(
        tmp_path,
        "from myapp.models import User\nUser.objects.annotate(em",
    )
    result = analyzer_basic.completions(uri, pos)
    assert result.exclusive is False
    assert result.items == []


def test_completion_unknown_receiver_silent(analyzer_basic, tmp_path):
    # We see a `filter(` but can't resolve the receiver — let ty handle.
    uri, pos = _write_with_cursor(
        tmp_path, "def f(qs):\n    qs.filter(em"
    )
    result = analyzer_basic.completions(uri, pos)
    assert result.exclusive is False
    assert result.items == []


def test_completion_relation_field_target(analyzer_blog, tmp_path):
    uri, pos = _write_with_cursor(
        tmp_path,
        "from blog.models import Article\nArticle.objects.filter(au",
    )
    result = analyzer_blog.completions(uri, pos)
    by_label = {it["label"]: it for it in result.items}
    assert "author" in by_label and "author_id" in by_label
    # detail shows the relation target on the field itself.
    assert "Author" in by_label["author"]["detail"]


def test_completion_reverse_relation_offered(analyzer_blog, tmp_path):
    uri, pos = _write_with_cursor(
        tmp_path,
        "from blog.models import Author\nAuthor.objects.filter(art",
    )
    result = analyzer_blog.completions(uri, pos)
    assert _labels(result) == ["articles"]
    assert "reverse" in result.items[0]["detail"]


def test_completion_fk_id_accessor_offered(analyzer_basic, tmp_path):
    uri, pos = _write_with_cursor(
        tmp_path,
        "from myapp.models import Profile\nProfile.objects.filter(user_",
    )
    labels = _labels(analyzer_basic.completions(uri, pos))
    assert "user_id" in labels


def test_completion_for_local_queryset(analyzer_basic, tmp_path):
    uri, pos = _write_with_cursor(
        tmp_path,
        "from myapp.models import User\n"
        "def f():\n"
        "    qs = User.objects.all()\n"
        "    qs.filter(em",
    )
    result = analyzer_basic.completions(uri, pos)
    assert result.exclusive is True
    assert _labels(result) == ["email"]


def test_completion_for_builtin_user(tmp_path: Path):
    # Workspace has no User; contrib stub kicks in.
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("")
    (tmp_path / "app" / "models.py").write_text(
        "from django.db import models\n"
        "class Review(models.Model):\n    text = models.TextField()\n"
    )
    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)

    src_path = tmp_path / "u.py"
    source = "from django.contrib.auth.models import User\nUser.objects.filter(em"
    src_path.write_text(source)
    uri = src_path.as_uri()
    result = a.completions(uri, {"line": 1, "character": 22})
    assert result.exclusive is True
    assert _labels(result) == ["email"]


def test_completion_partial_with_no_match_still_exclusive(analyzer_basic, tmp_path):
    # User typed `zzz` — no field starts with that. We still own this
    # position so ty's variable completions should be suppressed.
    uri, pos = _write_with_cursor(
        tmp_path, "from myapp.models import User\nUser.objects.filter(zzz"
    )
    result = analyzer_basic.completions(uri, pos)
    assert result.exclusive is True
    assert result.items == []


def test_completion_disabled_via_config(analyzer_basic, tmp_path):
    from dataclasses import replace
    analyzer_basic.config = replace(
        analyzer_basic.config, disabled_rules=frozenset({"orm_lookup"})
    )
    uri, pos = _write_with_cursor(
        tmp_path, "from myapp.models import User\nUser.objects.filter(em"
    )
    result = analyzer_basic.completions(uri, pos)
    assert result.exclusive is False
    assert result.items == []


def test_completion_text_provider_used(analyzer_basic, tmp_path):
    # Disk has a closed call; live buffer has the partial.
    src_path = tmp_path / "u.py"
    src_path.write_text(
        "from myapp.models import User\nUser.objects.filter(email='x')\n"
    )
    uri = src_path.as_uri()
    buffers: dict[str, str] = {
        uri: "from myapp.models import User\nUser.objects.filter(em"
    }
    analyzer_basic._text_provider = buffers.get

    result = analyzer_basic.completions(uri, {"line": 1, "character": 22})
    assert _labels(result) == ["email"]
