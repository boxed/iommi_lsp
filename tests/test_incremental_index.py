"""Tests for incremental file-change updates of the Django index."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

import pytest

from iommi_lsp.analyzers.django import (
    DjangoAnalyzer,
    assemble_index,
    collect_scrapes,
    update_scrapes,
)


def _write(path: Path, src: str) -> None:
    path.write_text(textwrap.dedent(src).lstrip())


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    pkg = tmp_path / "shop"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    _write(pkg / "models.py", """
        from django.db import models

        class Item(models.Model):
            name = models.CharField(max_length=80)
    """)
    return tmp_path


def test_collect_then_assemble_matches_build_index(workspace: Path):
    from iommi_lsp.analyzers.django import build_index
    expected = build_index(workspace)
    incremental = assemble_index(workspace, collect_scrapes(workspace))
    assert set(incremental.models) == set(expected.models)


def test_update_scrapes_picks_up_new_model(workspace: Path):
    scrapes = collect_scrapes(workspace)
    idx = assemble_index(workspace, scrapes)
    assert "shop.models.Order" not in idx.models

    _write(workspace / "shop" / "models.py", """
        from django.db import models

        class Item(models.Model):
            name = models.CharField(max_length=80)

        class Order(models.Model):
            item = models.ForeignKey(Item, on_delete=models.CASCADE)
    """)
    update_scrapes(workspace, scrapes, workspace / "shop" / "models.py")
    idx = assemble_index(workspace, scrapes)

    assert "shop.models.Order" in idx.models
    # The new FK creates a default reverse on Item.
    assert "order_set" in idx.reverse_relations["shop.models.Item"]


def test_update_scrapes_drops_removed_file(workspace: Path):
    scrapes = collect_scrapes(workspace)
    assert any("shop/models.py" in str(p) for p in scrapes)

    (workspace / "shop" / "models.py").unlink()
    update_scrapes(workspace, scrapes, workspace / "shop" / "models.py")

    idx = assemble_index(workspace, scrapes)
    assert "shop.models.Item" not in idx.models


def test_analyzer_on_file_changed_picks_up_new_model(workspace: Path):
    analyzer = DjangoAnalyzer(workspace_root=workspace)
    asyncio.get_event_loop_policy()
    asyncio.run(analyzer.index(workspace))
    assert "shop.models.Order" not in analyzer.django_index.models

    _write(workspace / "shop" / "models.py", """
        from django.db import models

        class Item(models.Model):
            name = models.CharField(max_length=80)

        class Order(models.Model):
            item = models.ForeignKey(Item, on_delete=models.CASCADE, related_name="orders")
    """)
    asyncio.run(analyzer.on_file_changed((workspace / "shop" / "models.py").as_uri()))

    assert "shop.models.Order" in analyzer.django_index.models
    assert "orders" in analyzer.django_index.reverse_relations["shop.models.Item"]


def test_analyzer_incremental_does_not_re_walk_workspace(
    workspace: Path, monkeypatch
):
    """on_file_changed must reuse the scrape cache — full re-walk is the
    M7 regression we're guarding against. We assert by counting calls."""
    from iommi_lsp.analyzers.django import index as index_mod

    analyzer = DjangoAnalyzer(workspace_root=workspace)
    asyncio.run(analyzer.index(workspace))

    calls = {"collect": 0, "scrape_one": 0}
    original_collect = index_mod.collect_scrapes
    original_scrape = index_mod.scrape_file

    def counting_collect(*a, **kw):
        calls["collect"] += 1
        return original_collect(*a, **kw)

    def counting_scrape(*a, **kw):
        calls["scrape_one"] += 1
        return original_scrape(*a, **kw)

    monkeypatch.setattr(index_mod, "collect_scrapes", counting_collect)
    monkeypatch.setattr(index_mod, "scrape_file", counting_scrape)

    asyncio.run(analyzer.on_file_changed((workspace / "shop" / "models.py").as_uri()))

    assert calls["collect"] == 0, "on_file_changed must not re-walk the workspace"
    assert calls["scrape_one"] == 1, "on_file_changed must re-parse exactly one file"
