"""Tests for ``[tool.iommi_lsp]`` config loading and rule gating."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

import pytest

from iommi_lsp import config as cfg_mod
from iommi_lsp.analyzers.django import DjangoAnalyzer, build_index
from iommi_lsp.config import DEFAULT, Config, load


def _write_pyproject(workspace: Path, body: str) -> None:
    (workspace / "pyproject.toml").write_text(textwrap.dedent(body).lstrip())


def test_no_pyproject_returns_default(tmp_path: Path):
    assert load(tmp_path) is DEFAULT


def test_no_iommi_lsp_section_returns_default(tmp_path: Path):
    _write_pyproject(tmp_path, """
        [project]
        name = "x"
    """)
    assert load(tmp_path) is DEFAULT


def test_full_config_round_trip(tmp_path: Path):
    _write_pyproject(tmp_path, """
        [tool.iommi_lsp]
        enabled = true
        disabled_rules = ["pk", "reverse"]

        [tool.iommi_lsp.extra_magic_attrs]
        manager = ["mongo", "search"]
    """)
    c = load(tmp_path)
    assert c.enabled is True
    assert c.disabled_rules == frozenset({"pk", "reverse"})
    assert c.extra_magic_attrs == {"manager": frozenset({"mongo", "search"})}


def test_unknown_rule_in_disabled_is_ignored(tmp_path: Path, caplog):
    _write_pyproject(tmp_path, """
        [tool.iommi_lsp]
        disabled_rules = ["pk", "no_such_rule"]
    """)
    c = load(tmp_path)
    assert c.disabled_rules == frozenset({"pk"})


def test_malformed_pyproject_falls_back_to_default(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("this is not [valid")
    assert load(tmp_path) is DEFAULT


def test_disabled_master_switch_returns_default_for_rule_check():
    c = Config(enabled=False)
    assert c.is_rule_enabled("manager") is False
    assert c.is_rule_enabled("anything") is False


def test_merged_static_attrs_combines_base_with_extra():
    c = Config(extra_magic_attrs={"manager": frozenset({"mongo"})})
    merged = c.merged_static_attrs("manager")
    assert "objects" in merged          # from MANAGER_ATTRS
    assert "mongo" in merged            # from config
    assert "_meta" not in merged        # different group


# ---------------------------------------------------------------------------
# Integration: DjangoAnalyzer should honor the config.
# ---------------------------------------------------------------------------


CORPUS = Path(__file__).parent / "corpus"


def _diag(line, col_start, col_end, attr):
    return {
        "code": "unresolved-attribute",
        "message": f"no attribute {attr!r}",
        "range": {
            "start": {"line": line, "character": col_start},
            "end": {"line": line, "character": col_end},
        },
        "severity": 1,
        "source": "ty",
    }


def test_disabled_master_switch_keeps_all_diagnostics(tmp_path: Path):
    src = "from myapp.models import User\n\ndef f():\n    return User.objects\n"
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(
        workspace_root=CORPUS / "basic_django",
        config=Config(enabled=False),
    )
    a.django_index = build_index(CORPUS / "basic_django")
    line = 3
    start = src.splitlines()[line].index("objects")
    diag = _diag(line, start, start + len("objects"), "objects")

    assert a.is_false_positive(f.as_uri(), diag) is False


def test_disabled_manager_rule_keeps_objects_diagnostic(tmp_path: Path):
    src = "from myapp.models import User\n\ndef f():\n    return User.objects\n"
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(
        workspace_root=CORPUS / "basic_django",
        config=Config(disabled_rules=frozenset({"manager"})),
    )
    a.django_index = build_index(CORPUS / "basic_django")
    line = 3
    start = src.splitlines()[line].index("objects")
    diag = _diag(line, start, start + len("objects"), "objects")

    assert a.is_false_positive(f.as_uri(), diag) is False


def test_disabled_manager_rule_still_filters_meta(tmp_path: Path):
    src = "from myapp.models import User\n\ndef f():\n    return User._meta\n"
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(
        workspace_root=CORPUS / "basic_django",
        config=Config(disabled_rules=frozenset({"manager"})),
    )
    a.django_index = build_index(CORPUS / "basic_django")
    line = 3
    start = src.splitlines()[line].index("_meta")
    diag = _diag(line, start, start + len("_meta"), "_meta")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_extra_magic_attrs_filters_custom_manager(tmp_path: Path):
    src = "from myapp.models import User\n\ndef f():\n    return User.mongo\n"
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(
        workspace_root=CORPUS / "basic_django",
        config=Config(extra_magic_attrs={"manager": frozenset({"mongo"})}),
    )
    a.django_index = build_index(CORPUS / "basic_django")
    line = 3
    start = src.splitlines()[line].index("mongo")
    diag = _diag(line, start, start + len("mongo"), "mongo")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_disabled_reverse_keeps_articles_diagnostic(tmp_path: Path):
    src = (
        "from blog.models import Author\n"
        "\n"
        "def f():\n"
        "    a = Author.objects.first()\n"
        "    return a.articles\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(
        workspace_root=CORPUS / "related_names",
        config=Config(disabled_rules=frozenset({"reverse"})),
    )
    a.django_index = build_index(CORPUS / "related_names")
    line = 4
    start = src.splitlines()[line].index("articles")
    diag = _diag(line, start, start + len("articles"), "articles")

    assert a.is_false_positive(f.as_uri(), diag) is False


def test_index_picks_up_config_from_pyproject(tmp_path: Path):
    """End-to-end: the analyzer's index() call should load config alongside."""
    pkg = tmp_path / "shop"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "models.py").write_text(
        "from django.db import models\n"
        "class Item(models.Model):\n"
        "    name = models.CharField(max_length=80)\n"
    )
    _write_pyproject(tmp_path, """
        [tool.iommi_lsp]
        disabled_rules = ["manager"]
    """)

    analyzer = DjangoAnalyzer(workspace_root=tmp_path)
    asyncio.run(analyzer.index(tmp_path))

    assert analyzer.config.disabled_rules == frozenset({"manager"})
    assert "shop.models.Item" in analyzer.django_index.models
