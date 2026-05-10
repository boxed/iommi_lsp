"""Tests for DjangoAnalyzer.additional_diagnostics — ORM ``__`` validation."""

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


def _write(tmp_path: Path, src: str) -> str:
    f = tmp_path / "u.py"
    f.write_text(src)
    return f.as_uri()


def test_valid_filter_emits_nothing(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from myapp.models import User\n"
        "User.objects.filter(username__icontains='a', email='b@c').first()\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.additional_diagnostics(uri) == []


def test_unknown_field_is_flagged(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from myapp.models import User\n"
        "User.objects.filter(bogus='a').first()\n"
    )
    uri = _write(tmp_path, src)
    diags = analyzer_basic.additional_diagnostics(uri)
    assert len(diags) == 1
    d = diags[0]
    assert d["code"] == "django-unknown-orm-lookup"
    assert d["data"]["outcome"] == "unknown_field"
    assert "bogus" in d["message"]
    # Range should pin the bad segment.
    line_text = src.splitlines()[d["range"]["start"]["line"]]
    assert line_text[d["range"]["start"]["character"]:d["range"]["end"]["character"]] == "bogus"


def test_unknown_lookup_is_flagged(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from myapp.models import User\n"
        "User.objects.filter(username__notalookup='x').first()\n"
    )
    uri = _write(tmp_path, src)
    diags = analyzer_basic.additional_diagnostics(uri)
    assert len(diags) == 1
    d = diags[0]
    assert d["data"]["outcome"] == "unknown_lookup"
    assert "notalookup" in d["message"]
    line_text = src.splitlines()[d["range"]["start"]["line"]]
    assert line_text[d["range"]["start"]["character"]:d["range"]["end"]["character"]] == "notalookup"


def test_relation_traversal_validates(analyzer_blog: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from blog.models import Article\n"
        "Article.objects.filter(author__name__icontains='x').first()\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_blog.additional_diagnostics(uri) == []


def test_relation_traversal_unknown_target_field(
    analyzer_blog: DjangoAnalyzer, tmp_path: Path
):
    src = (
        "from blog.models import Article\n"
        "Article.objects.filter(author__bogus='x').first()\n"
    )
    uri = _write(tmp_path, src)
    diags = analyzer_blog.additional_diagnostics(uri)
    assert len(diags) == 1
    assert diags[0]["data"]["outcome"] == "unknown_field"
    assert diags[0]["data"]["on_model"] == "blog.models.Author"


def test_reverse_relation_traversal(analyzer_blog: DjangoAnalyzer, tmp_path: Path):
    # Author has reverse `articles` (related_name) to Article.
    src = (
        "from blog.models import Author\n"
        "Author.objects.filter(articles__title__startswith='x').first()\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_blog.additional_diagnostics(uri) == []


def test_chained_filter_exclude(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from myapp.models import User\n"
        "User.objects.filter(username='a').exclude(bogus=1).first()\n"
    )
    uri = _write(tmp_path, src)
    diags = analyzer_basic.additional_diagnostics(uri)
    # Only the bogus kwarg should be flagged.
    assert len(diags) == 1
    assert "bogus" in diags[0]["message"]


def test_pk_is_accepted(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from myapp.models import User\n"
        "User.objects.filter(pk=1).first()\n"
        "User.objects.filter(pk__in=[1, 2]).first()\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.additional_diagnostics(uri) == []


def test_fk_id_accessor_is_accepted(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from myapp.models import Profile\n"
        "Profile.objects.filter(user_id=1).first()\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.additional_diagnostics(uri) == []


def test_unknown_receiver_is_silent(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    # `qs` is some local — we don't try to resolve, so we say nothing.
    src = (
        "def f(qs):\n"
        "    qs.filter(literally_anything=1).first()\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.additional_diagnostics(uri) == []


def test_get_or_create_defaults_kwarg_is_skipped(
    analyzer_basic: DjangoAnalyzer, tmp_path: Path
):
    src = (
        "from myapp.models import User\n"
        "User.objects.get_or_create(username='a', defaults={'email': 'x'})\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.additional_diagnostics(uri) == []


def test_q_object_valid_kwargs(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.db.models import Q\n"
        "from myapp.models import User\n"
        "User.objects.filter(Q(username__icontains='a') | Q(email='b'))\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.additional_diagnostics(uri) == []


def test_q_object_unknown_field(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.db.models import Q\n"
        "from myapp.models import User\n"
        "User.objects.filter(Q(bogus='a'))\n"
    )
    uri = _write(tmp_path, src)
    diags = analyzer_basic.additional_diagnostics(uri)
    assert len(diags) == 1
    assert diags[0]["data"]["outcome"] == "unknown_field"
    assert "bogus" in diags[0]["message"]


def test_q_unknown_field_inside_or(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.db.models import Q\n"
        "from myapp.models import User\n"
        "User.objects.filter(Q(username='a') | Q(bogus='b') | ~Q(email='c'))\n"
    )
    uri = _write(tmp_path, src)
    diags = analyzer_basic.additional_diagnostics(uri)
    assert len(diags) == 1
    assert "bogus" in diags[0]["message"]


def test_q_with_models_dot_q(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.db import models\n"
        "from myapp.models import User\n"
        "User.objects.filter(models.Q(bogus='a'))\n"
    )
    uri = _write(tmp_path, src)
    diags = analyzer_basic.additional_diagnostics(uri)
    assert len(diags) == 1
    assert "bogus" in diags[0]["message"]


def test_q_mixed_with_kwargs(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.db.models import Q\n"
        "from myapp.models import User\n"
        "User.objects.filter(Q(bogus_q='a'), bogus_kw='b')\n"
    )
    uri = _write(tmp_path, src)
    diags = analyzer_basic.additional_diagnostics(uri)
    bad = sorted(d["range"]["start"]["character"] for d in diags)
    msgs = [d["message"] for d in diags]
    assert len(diags) == 2
    assert any("bogus_q" in m for m in msgs)
    assert any("bogus_kw" in m for m in msgs)


def test_q_relation_traversal(analyzer_blog: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.db.models import Q\n"
        "from blog.models import Article\n"
        "Article.objects.filter(Q(author__name__icontains='x'))\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_blog.additional_diagnostics(uri) == []


def test_order_by_valid(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from myapp.models import User\n"
        "User.objects.order_by('username', '-email', 'pk')\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.additional_diagnostics(uri) == []


def test_order_by_random_token(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from myapp.models import User\n"
        "User.objects.order_by('?')\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.additional_diagnostics(uri) == []


def test_order_by_unknown_field_with_dash(
    analyzer_basic: DjangoAnalyzer, tmp_path: Path
):
    src = (
        "from myapp.models import User\n"
        "User.objects.order_by('-bogus')\n"
    )
    uri = _write(tmp_path, src)
    diags = analyzer_basic.additional_diagnostics(uri)
    assert len(diags) == 1
    d = diags[0]
    assert d["data"]["outcome"] == "unknown_field"
    line = src.splitlines()[d["range"]["start"]["line"]]
    pinned = line[d["range"]["start"]["character"]:d["range"]["end"]["character"]]
    assert pinned == "bogus"   # not "-bogus"


def test_order_by_relation_traversal(
    analyzer_blog: DjangoAnalyzer, tmp_path: Path
):
    src = (
        "from blog.models import Article\n"
        "Article.objects.order_by('author__name', '-author__bogus')\n"
    )
    uri = _write(tmp_path, src)
    diags = analyzer_blog.additional_diagnostics(uri)
    assert len(diags) == 1
    assert diags[0]["data"]["on_model"] == "blog.models.Author"


def test_values_only_defer_distinct(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from myapp.models import User\n"
        "User.objects.values('username').only('email').defer('bogus').distinct()\n"
    )
    uri = _write(tmp_path, src)
    diags = analyzer_basic.additional_diagnostics(uri)
    assert len(diags) == 1
    assert "bogus" in diags[0]["message"]


def test_values_list_with_flat_kwarg(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    # Positional strings are field paths; `flat=True` is a kwarg, not validated.
    src = (
        "from myapp.models import User\n"
        "User.objects.values_list('username', flat=True)\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.additional_diagnostics(uri) == []


def test_select_related_relation_path(
    analyzer_blog: DjangoAnalyzer, tmp_path: Path
):
    src = (
        "from blog.models import Article\n"
        "Article.objects.select_related('author').filter(title='x')\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_blog.additional_diagnostics(uri) == []


def test_select_related_unknown(analyzer_blog: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from blog.models import Article\n"
        "Article.objects.select_related('bogus')\n"
    )
    uri = _write(tmp_path, src)
    diags = analyzer_blog.additional_diagnostics(uri)
    assert len(diags) == 1
    assert "bogus" in diags[0]["message"]


def test_prefetch_related_reverse(analyzer_blog: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from blog.models import Author\n"
        "Author.objects.prefetch_related('articles')\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_blog.additional_diagnostics(uri) == []


def test_prefetch_related_skips_non_string_args(
    analyzer_blog: DjangoAnalyzer, tmp_path: Path
):
    # `Prefetch(...)` objects pass through silently.
    src = (
        "from django.db.models import Prefetch\n"
        "from blog.models import Author\n"
        "Author.objects.prefetch_related(Prefetch('articles'))\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_blog.additional_diagnostics(uri) == []


def test_update_kwargs(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from myapp.models import User\n"
        "User.objects.filter(pk=1).update(username='x', bogus='y')\n"
    )
    uri = _write(tmp_path, src)
    diags = analyzer_basic.additional_diagnostics(uri)
    assert len(diags) == 1
    assert "bogus" in diags[0]["message"]


def test_create_kwargs(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from myapp.models import User\n"
        "User.objects.create(username='x', bogus=1)\n"
    )
    uri = _write(tmp_path, src)
    diags = analyzer_basic.additional_diagnostics(uri)
    assert len(diags) == 1
    assert "bogus" in diags[0]["message"]


def test_f_expression_valid(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.db.models import F\n"
        "from myapp.models import User\n"
        "User.objects.update(username=F('email'))\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.additional_diagnostics(uri) == []


def test_f_expression_unknown(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.db.models import F\n"
        "from myapp.models import User\n"
        "User.objects.update(username=F('bogus'))\n"
    )
    uri = _write(tmp_path, src)
    diags = analyzer_basic.additional_diagnostics(uri)
    assert len(diags) == 1
    assert "bogus" in diags[0]["message"]


def test_f_expression_relation_path(analyzer_blog: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.db.models import F\n"
        "from blog.models import Article\n"
        "Article.objects.filter(title=F('author__name'))\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_blog.additional_diagnostics(uri) == []


def test_f_inside_q(analyzer_blog: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.db.models import F, Q\n"
        "from blog.models import Article\n"
        "Article.objects.filter(Q(title=F('author__bogus')))\n"
    )
    uri = _write(tmp_path, src)
    diags = analyzer_blog.additional_diagnostics(uri)
    assert len(diags) == 1
    assert diags[0]["data"]["on_model"] == "blog.models.Author"


def test_disabled_via_config(analyzer_basic: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from myapp.models import User\n"
        "User.objects.filter(bogus=1).first()\n"
    )
    uri = _write(tmp_path, src)
    # Disable the rule and verify no diagnostics emitted.
    from dataclasses import replace
    analyzer_basic.config = replace(
        analyzer_basic.config,
        disabled_rules=frozenset({"orm_lookup"}),
    )
    assert analyzer_basic.additional_diagnostics(uri) == []
