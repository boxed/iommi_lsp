"""Unit tests for DjangoAnalyzer.is_false_positive."""

from __future__ import annotations

from pathlib import Path

import pytest

from iommi_lsp.analyzers.django import DjangoAnalyzer, build_index


CORPUS = Path(__file__).parent / "corpus"


def _diag(line: int, col_start: int, col_end: int, attr: str, code: str = "unresolved-attribute"):
    return {
        "code": code,
        "message": f"Type \"…\" has no attribute \"{attr}\"",
        "range": {
            "start": {"line": line, "character": col_start},
            "end": {"line": line, "character": col_end},
        },
        "severity": 1,
        "source": "ty",
    }


@pytest.fixture
def analyzer() -> DjangoAnalyzer:
    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")
    return a


def _uri_for(rel_path: str) -> str:
    return (CORPUS / rel_path).as_uri()


def test_objects_on_known_model_is_dropped(tmp_path: Path):
    src = "from myapp.models import User\n\ndef f():\n    return User.objects\n"
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    # Diagnostic on `objects` (line 3, "    return User.objects\n").
    # `objects` starts after `User.` at index 4 + len("return User.") = 4 + 12 = 16
    line = 3
    start = src.splitlines()[line].index("objects")
    diag = _diag(line, start, start + len("objects"), "objects")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_unknown_model_is_kept(tmp_path: Path):
    src = "class Foo:\n    pass\n\ndef f():\n    return Foo.objects\n"
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 4
    start = src.splitlines()[line].index("objects")
    diag = _diag(line, start, start + len("objects"), "objects")

    assert a.is_false_positive(f.as_uri(), diag) is False


def test_non_unresolved_attribute_diagnostics_are_ignored(tmp_path: Path):
    src = "from myapp.models import User\n\ndef f():\n    return User.objects\n"
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    start = src.splitlines()[line].index("objects")
    diag = _diag(line, start, start + len("objects"), "objects", code="some-other-rule")

    assert a.is_false_positive(f.as_uri(), diag) is False


def test_local_flow_assignment_resolves_receiver(tmp_path: Path):
    src = (
        "from myapp.models import User\n"
        "\n"
        "def fetch():\n"
        "    user = User.objects.get(pk=1)\n"
        "    return user.pk\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 4  # "    return user.pk"
    start = src.splitlines()[line].index("pk")
    diag = _diag(line, start, start + len("pk"), "pk")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_id_kept_when_explicit_pk_declared(tmp_path: Path):
    src = (
        "from myapp.models import WithExplicitPK\n"
        "\n"
        "def f():\n"
        "    return WithExplicitPK.id\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    text = src.splitlines()[line]
    start = text.rindex("id")
    diag = _diag(line, start, start + 2, "id")

    # Explicit PK -> `id` is NOT auto-injected, so we should NOT suppress.
    assert a.is_false_positive(f.as_uri(), diag) is False


def test_pk_kept_for_explicit_pk_model(tmp_path: Path):
    """`pk` is still always present even with explicit PK — must drop."""
    src = (
        "from myapp.models import WithExplicitPK\n"
        "\n"
        "def f():\n"
        "    return WithExplicitPK.pk\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    text = src.splitlines()[line]
    start = text.rindex("pk")
    diag = _diag(line, start, start + 2, "pk")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_reverse_relation_is_dropped(tmp_path: Path):
    src = (
        "from blog.models import Author\n"
        "\n"
        "def f():\n"
        "    a = Author.objects.first()\n"
        "    return a.articles\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "related_names")
    a.django_index = build_index(CORPUS / "related_names")

    line = 4
    start = src.splitlines()[line].index("articles")
    diag = _diag(line, start, start + len("articles"), "articles")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_default_reverse_set_is_dropped(tmp_path: Path):
    src = (
        "from blog.models import Article\n"
        "\n"
        "def f():\n"
        "    a = Article.objects.first()\n"
        "    return a.comment_set\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "related_names")
    a.django_index = build_index(CORPUS / "related_names")

    line = 4
    start = src.splitlines()[line].index("comment_set")
    diag = _diag(line, start, start + len("comment_set"), "comment_set")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_fk_id_accessor_is_dropped(tmp_path: Path):
    src = (
        "from myapp.models import Profile\n"
        "\n"
        "def f():\n"
        "    return Profile.user_id\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    start = src.splitlines()[line].index("user_id")
    diag = _diag(line, start, start + len("user_id"), "user_id")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_unrelated_attribute_is_kept(tmp_path: Path):
    """Genuine unresolved attribute on a model must NOT be suppressed."""
    src = (
        "from myapp.models import User\n"
        "\n"
        "def f():\n"
        "    return User.objects_typo\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    start = src.splitlines()[line].index("objects_typo")
    diag = _diag(line, start, start + len("objects_typo"), "objects_typo")

    assert a.is_false_positive(f.as_uri(), diag) is False
