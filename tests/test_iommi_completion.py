"""Tests for IommiAnalyzer.completions — refinable-kwarg completion items."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from iommi_lsp.analyzers.iommi import IommiAnalyzer
from iommi_lsp.analyzers.iommi.graph import (
    GRAPH_FILENAME,
    IommiClass,
    IommiGraph,
    Refinable,
    save_graph,
)


def _r(name, kind, **kw):
    return Refinable(name=name, kind=kind, **kw)


def _make_fixture_graph() -> IommiGraph:
    column = IommiClass(
        qualname="iommi.table.Column",
        bases=["iommi.part.Part"],
        refinables={
            "extra": _r("extra", "open_namespace"),
            "after": _r("after", "evaluated_scalar"),
            "cell": _r("cell", "namespace", known_keys=["attrs", "contents"]),
        },
    )
    part = IommiClass(
        qualname="iommi.part.Part",
        bases=[],
        refinables={
            "attrs": _r(
                "attrs", "html_attrs",
                sub_specials={
                    "class": {"value_type": "bool"},
                    "style": {"value_type": "str"},
                },
            ),
        },
    )
    form = IommiClass(
        qualname="iommi.form.Form",
        bases=["iommi.part.Part"],
        refinables={
            "fields": _r("fields", "members"),
            "title": _r("title", "evaluated_scalar"),
        },
    )
    table = IommiClass(
        qualname="iommi.table.Table",
        bases=["iommi.part.Part"],
        refinables={
            "columns": _r("columns", "members", member_class="iommi.table.Column"),
            "page_size": _r("page_size", "evaluated_scalar"),
            "bulk": _r("bulk", "class_ref", target="iommi.form.Form"),
            "extra": _r("extra", "open_namespace"),
            "attrs": _r(
                "attrs", "html_attrs",
                sub_specials={
                    "class": {"value_type": "bool"},
                    "style": {"value_type": "str"},
                },
            ),
            "auto": _r(
                "auto", "namespace",
                known_keys=["model", "rows", "instance", "include", "exclude"],
            ),
        },
    )
    return IommiGraph(
        iommi_version="0.0-test",
        classes={c.qualname: c for c in [table, column, part, form]},
    )


@pytest.fixture
def analyzer(tmp_path: Path) -> IommiAnalyzer:
    save_graph(_make_fixture_graph(), tmp_path / GRAPH_FILENAME)
    a = IommiAnalyzer(workspace_root=tmp_path)
    asyncio.run(a.index(tmp_path))
    return a


@pytest.fixture
def analyzer_with_django(tmp_path: Path) -> IommiAnalyzer:
    """iommi analyzer wired to a Django index containing User/Article."""
    from iommi_lsp.analyzers.django.index import (
        DjangoIndex,
        FieldInfo,
        ModelInfo,
    )

    save_graph(_make_fixture_graph(), tmp_path / GRAPH_FILENAME)

    user = ModelInfo(
        qualname="myapp.models.User",
        module="myapp.models",
        name="User",
        file_path=tmp_path / "myapp" / "models.py",
        bases=["django.db.models.Model"],
        fields={
            "username": FieldInfo(name="username", field_type="CharField"),
            "email": FieldInfo(name="email", field_type="EmailField"),
        },
    )
    article = ModelInfo(
        qualname="blog.models.Article",
        module="blog.models",
        name="Article",
        file_path=tmp_path / "blog" / "models.py",
        bases=["django.db.models.Model"],
        fields={
            "title": FieldInfo(name="title", field_type="CharField"),
        },
    )
    index = DjangoIndex()
    index.add_model(user)
    index.add_model(article)

    a = IommiAnalyzer(
        workspace_root=tmp_path,
        django_index_provider=lambda: index,
    )
    asyncio.run(a.index(tmp_path))
    return a


def _write_with_cursor(
    tmp_path: Path, src_before: str, src_after: str = ""
) -> tuple[str, dict]:
    f = tmp_path / "u.py"
    f.write_text(src_before + src_after)
    line = src_before.count("\n")
    last_nl = src_before.rfind("\n")
    character = len(src_before) - (last_nl + 1)
    return f.as_uri(), {"line": line, "character": character}


def _labels(result) -> list[str]:
    return [it["label"] for it in result.items]


def test_top_level_empty_partial_returns_all_refinables(analyzer, tmp_path):
    uri, pos = _write_with_cursor(tmp_path, "from iommi import Table\nTable(")
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    labels = set(_labels(result))
    # Container refinables — drill in.
    assert "columns__" in labels
    assert "attrs__" in labels
    # Scalar — gets `=`.
    assert "page_size" in labels


def test_top_level_partial_prefix_filters(analyzer, tmp_path):
    uri, pos = _write_with_cursor(tmp_path, "from iommi import Table\nTable(co")
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    # `columns` is a members refinable → trailing `__`.
    assert _labels(result) == ["columns__"]
    assert result.items[0]["insertText"] == "columns__"


def test_no_match_still_exclusive(analyzer, tmp_path):
    uri, pos = _write_with_cursor(tmp_path, "from iommi import Table\nTable(zzz")
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    assert result.items == []


def test_chain_into_members_member_class(analyzer, tmp_path):
    # `columns` is a members refinable; after the column name, we're inside
    # Column — suggest its refinables.
    uri, pos = _write_with_cursor(
        tmp_path, "from iommi import Table\nTable(columns__name__"
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    labels = set(_labels(result))
    # Scalar → no trailing.
    assert "columns__name__after" in labels
    # Containers → trailing `__`.
    assert "columns__name__cell__" in labels
    assert "columns__name__attrs__" in labels   # inherited from Part


def test_chain_into_members_with_partial(analyzer, tmp_path):
    uri, pos = _write_with_cursor(
        tmp_path, "from iommi import Table\nTable(columns__name__af"
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    assert _labels(result) == ["columns__name__after"]
    assert result.items[0]["insertText"] == "columns__name__after="


def test_member_name_slot_is_silent(analyzer, tmp_path):
    # `columns__` puts us at the member-name slot (user picks any name).
    # We can't enumerate — exclusive empty (still ours, no useful items).
    uri, pos = _write_with_cursor(
        tmp_path, "from iommi import Table\nTable(columns__"
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    assert result.items == []


def test_chain_through_class_ref(analyzer, tmp_path):
    # `bulk` is a class_ref to Form — chain through it.
    uri, pos = _write_with_cursor(
        tmp_path, "from iommi import Table\nTable(bulk__"
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    labels = set(_labels(result))
    assert "bulk__fields__" in labels   # members → drill
    assert "bulk__title" in labels      # scalar
    assert "bulk__attrs__" in labels    # html_attrs via Part → drill


def test_namespace_keys_suggested(analyzer, tmp_path):
    # Drop to a Column-level namespace: `columns__name__cell__`.
    uri, pos = _write_with_cursor(
        tmp_path,
        "from iommi import Table\nTable(columns__name__cell__",
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    labels = set(_labels(result))
    assert "columns__name__cell__attrs__" in labels   # attrs → html_attrs, drill
    assert "columns__name__cell__contents" in labels   # default scalar


def test_html_attrs_offers_class_and_style(analyzer, tmp_path):
    uri, pos = _write_with_cursor(
        tmp_path, "from iommi import Table\nTable(attrs__"
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    labels = set(_labels(result))
    # `class` / `style` are dict-keyed; drill in.
    assert "attrs__class__" in labels
    assert "attrs__style__" in labels


def test_html_attrs_chain_into_class_no_enumeration(analyzer, tmp_path):
    # `attrs__class__` — user supplies arbitrary class name keys. We don't
    # know them, so exclusive empty (we own the slot).
    uri, pos = _write_with_cursor(
        tmp_path, "from iommi import Table\nTable(attrs__class__"
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    assert result.items == []


def test_chain_past_scalar_silent(analyzer, tmp_path):
    # `page_size` is a leaf. `page_size__` is invalid; offer nothing.
    uri, pos = _write_with_cursor(
        tmp_path, "from iommi import Table\nTable(page_size__"
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    assert result.items == []


def test_open_namespace_is_silent(analyzer, tmp_path):
    # `extra__` accepts anything — we can't enumerate.
    uri, pos = _write_with_cursor(
        tmp_path, "from iommi import Table\nTable(extra__"
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    assert result.items == []


def test_unknown_class_silent(analyzer, tmp_path):
    # `NotIommi` isn't in the graph — return non-exclusive empty so ty fills in.
    uri, pos = _write_with_cursor(
        tmp_path,
        "class NotIommi:\n    def __init__(self, **kw): pass\nNotIommi(",
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is False
    assert result.items == []


def test_unknown_segment_silent(analyzer, tmp_path):
    uri, pos = _write_with_cursor(
        tmp_path, "from iommi import Table\nTable(bogus__"
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    assert result.items == []


def test_module_qualified_call(analyzer, tmp_path):
    uri, pos = _write_with_cursor(
        tmp_path, "import iommi\niommi.Table(co"
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    assert _labels(result) == ["columns__"]


def test_no_graph_still_recognizes_iommi_classes(tmp_path):
    # Without `iommi_lsp graph build`, we still synthesise a stub for
    # the well-known iommi classes so the user gets exclusive completions
    # and ty's variable noise stays out.
    a = IommiAnalyzer(workspace_root=tmp_path, auto_build=False)
    asyncio.run(a.index(tmp_path))   # no graph file + no auto-build → empty graph
    uri, pos = _write_with_cursor(tmp_path, "from iommi import Table\nTable(co")
    result = a.completions(uri, pos)
    assert result.exclusive is True
    assert _labels(result) == ["columns__"]


def test_no_graph_auto_chain_works(tmp_path):
    a = IommiAnalyzer(workspace_root=tmp_path, auto_build=False)
    asyncio.run(a.index(tmp_path))
    uri, pos = _write_with_cursor(
        tmp_path, "from iommi import Table\nTable(auto__"
    )
    result = a.completions(uri, pos)
    assert result.exclusive is True
    labels = set(_labels(result))
    assert {"auto__model", "auto__rows", "auto__instance",
            "auto__include", "auto__exclude"} <= labels


def test_no_graph_auto_chain_partial(tmp_path):
    a = IommiAnalyzer(workspace_root=tmp_path, auto_build=False)
    asyncio.run(a.index(tmp_path))
    uri, pos = _write_with_cursor(
        tmp_path, "from iommi import Table\nTable(auto__mo"
    )
    result = a.completions(uri, pos)
    assert result.exclusive is True
    assert _labels(result) == ["auto__model"]


def test_no_graph_columns_with_auto_model_offers_fields(tmp_path):
    """No graph + Django index → still offer User's fields after auto__model."""
    from iommi_lsp.analyzers.django.index import (
        DjangoIndex, FieldInfo, ModelInfo,
    )
    user = ModelInfo(
        qualname="myapp.models.User", module="myapp.models", name="User",
        file_path=tmp_path / "u.py", bases=["django.db.models.Model"],
        fields={
            "username": FieldInfo(name="username", field_type="CharField"),
            "email": FieldInfo(name="email", field_type="EmailField"),
        },
    )
    index = DjangoIndex()
    index.add_model(user)

    a = IommiAnalyzer(
        workspace_root=tmp_path,
        django_index_provider=lambda: index,
        auto_build=False,
    )
    asyncio.run(a.index(tmp_path))
    uri, pos = _write_with_cursor(
        tmp_path,
        "from iommi import Table\nfrom myapp.models import User\n"
        "Table(auto__model=User, columns__",
    )
    result = a.completions(uri, pos)
    assert result.exclusive is True
    labels = set(_labels(result))
    assert "columns__username__" in labels
    assert "columns__email__" in labels


def test_no_graph_form_synthesised(tmp_path):
    a = IommiAnalyzer(workspace_root=tmp_path, auto_build=False)
    asyncio.run(a.index(tmp_path))
    uri, pos = _write_with_cursor(tmp_path, "from iommi import Form\nForm(fi")
    result = a.completions(uri, pos)
    assert result.exclusive is True
    assert _labels(result) == ["fields__"]


def test_no_graph_unknown_callable_passes_through(tmp_path):
    # An unfamiliar Capital-Case callable isn't synthesised — we don't
    # claim positions in arbitrary user code.
    a = IommiAnalyzer(workspace_root=tmp_path, auto_build=False)
    asyncio.run(a.index(tmp_path))
    uri, pos = _write_with_cursor(tmp_path, "MyThing(co")
    result = a.completions(uri, pos)
    assert result.exclusive is False
    assert result.items == []


# ---------------------------------------------------------------------------
# traditional_class — Column.cell drilling into Cell.__init__'s members
# ---------------------------------------------------------------------------


def _make_traditional_graph() -> IommiGraph:
    cell = IommiClass(
        qualname="iommi.table.Cell",
        bases=[],
        refinables={},
        init_members=["url", "url_title", "value", "tag", "contents"],
    )
    column = IommiClass(
        qualname="iommi.table.Column",
        bases=[],
        refinables={
            "cell": _r("cell", "traditional_class", target="iommi.table.Cell"),
            "after": _r("after", "evaluated_scalar"),
        },
    )
    table = IommiClass(
        qualname="iommi.table.Table",
        bases=[],
        refinables={
            "columns": _r(
                "columns", "members", member_class="iommi.table.Column",
            ),
            "cell": _r("cell", "traditional_class", target="iommi.table.Cell"),
        },
    )
    return IommiGraph(
        iommi_version="0.0-test",
        classes={c.qualname: c for c in [table, column, cell]},
    )


@pytest.fixture
def traditional_analyzer(tmp_path: Path) -> IommiAnalyzer:
    save_graph(_make_traditional_graph(), tmp_path / GRAPH_FILENAME)
    a = IommiAnalyzer(workspace_root=tmp_path)
    asyncio.run(a.index(tmp_path))
    return a


def test_traditional_class_offers_init_members(traditional_analyzer, tmp_path):
    """``Column(cell__`` lists Cell.__init__ attrs as leaf completions."""
    uri, pos = _write_with_cursor(
        tmp_path,
        "from iommi import Table\nTable(columns__name__cell__",
    )
    result = traditional_analyzer.completions(uri, pos)
    assert result.exclusive is True
    labels = set(_labels(result))
    assert "columns__name__cell__url" in labels
    assert "columns__name__cell__url_title" in labels
    assert "columns__name__cell__value" in labels
    # Leaf — `=` suffix, no `__`.
    url_item = next(
        it for it in result.items if it["label"] == "columns__name__cell__url"
    )
    assert url_item["insertText"] == "columns__name__cell__url="


def test_traditional_class_filters_by_partial(traditional_analyzer, tmp_path):
    uri, pos = _write_with_cursor(
        tmp_path,
        "from iommi import Table\nTable(columns__name__cell__url",
    )
    result = traditional_analyzer.completions(uri, pos)
    assert result.exclusive is True
    labels = set(_labels(result))
    assert labels == {
        "columns__name__cell__url",
        "columns__name__cell__url_title",
    }


def test_traditional_class_at_top_level_on_table(traditional_analyzer, tmp_path):
    """Table.cell is also a traditional_class — same drill behaviour."""
    uri, pos = _write_with_cursor(
        tmp_path, "from iommi import Table\nTable(cell__",
    )
    result = traditional_analyzer.completions(uri, pos)
    assert result.exclusive is True
    labels = set(_labels(result))
    assert "cell__url" in labels
    assert "cell__value" in labels


def test_chain_past_traditional_class_leaf_is_empty(traditional_analyzer, tmp_path):
    """``cell__url__<anything>`` is invalid — no completions to offer."""
    uri, pos = _write_with_cursor(
        tmp_path,
        "from iommi import Table\nTable(columns__name__cell__url__",
    )
    result = traditional_analyzer.completions(uri, pos)
    assert result.exclusive is True
    assert result.items == []


def test_inside_chained_call_for_member_class_refinables(analyzer, tmp_path):
    # `columns__name__cell__attrs__` — html_attrs reached via a namespace key.
    uri, pos = _write_with_cursor(
        tmp_path,
        "from iommi import Table\nTable(columns__name__cell__attrs__",
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    labels = set(_labels(result))
    assert "columns__name__cell__attrs__class__" in labels
    assert "columns__name__cell__attrs__style__" in labels


# ---------------------------------------------------------------------------
# auto refinable: drill-in suffix
# ---------------------------------------------------------------------------


def test_auto_top_level_suggests_drill_in(analyzer, tmp_path):
    # `Table(au` → `auto__` (namespace → drill).
    uri, pos = _write_with_cursor(tmp_path, "from iommi import Table\nTable(au")
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    assert _labels(result) == ["auto__"]
    assert result.items[0]["insertText"] == "auto__"


def test_auto_partial_chain_suggests_known_keys(analyzer, tmp_path):
    # `Table(auto__mo` should suggest `auto__model` even when the real
    # iommi graph reflects `auto` as `open_namespace` (its default
    # `Namespace()` is empty). Without the synthetic fallback, ty's
    # variable noise would leak through here.
    uri, pos = _write_with_cursor(
        tmp_path, "from iommi import Table\nTable(auto__mo"
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    assert _labels(result) == ["auto__model"]


def test_auto_top_level_synthesized_when_missing_from_graph(tmp_path):
    # `Table.auto` not reflected (some iommi versions / older releases)
    # → we still synthesize `auto__` so `Table(au` doesn't fall through
    # to ty's variable noise.
    table = IommiClass(
        qualname="iommi.table.Table", bases=[],
        refinables={
            "columns": _r("columns", "members", member_class="iommi.table.Column"),
            "page_size": _r("page_size", "evaluated_scalar"),
        },
    )
    column = IommiClass(
        qualname="iommi.table.Column", bases=[],
        refinables={"after": _r("after", "evaluated_scalar")},
    )
    graph = IommiGraph(
        iommi_version="0.0-test",
        classes={c.qualname: c for c in [table, column]},
    )
    save_graph(graph, tmp_path / GRAPH_FILENAME)
    a = IommiAnalyzer(workspace_root=tmp_path)
    asyncio.run(a.index(tmp_path))

    uri, pos = _write_with_cursor(
        tmp_path, "from iommi import Table\nTable(au"
    )
    result = a.completions(uri, pos)
    assert result.exclusive is True
    assert _labels(result) == ["auto__"]
    assert result.items[0]["insertText"] == "auto__"


def test_auto_not_synthesized_for_classes_without_members(tmp_path):
    # `Column` has no members refinables → don't pretend it supports auto.
    column = IommiClass(
        qualname="iommi.table.Column", bases=[],
        refinables={"after": _r("after", "evaluated_scalar")},
    )
    graph = IommiGraph(
        iommi_version="0.0-test",
        classes={column.qualname: column},
    )
    save_graph(graph, tmp_path / GRAPH_FILENAME)
    a = IommiAnalyzer(workspace_root=tmp_path)
    asyncio.run(a.index(tmp_path))

    uri, pos = _write_with_cursor(
        tmp_path, "from iommi import Column\nColumn(au"
    )
    result = a.completions(uri, pos)
    assert result.exclusive is True
    # No `auto__` for non-members classes.
    assert "auto__" not in _labels(result)


def test_auto_open_namespace_in_graph_still_suggests_keys(tmp_path):
    # Same as above but the graph reflects `auto` as open_namespace.
    column = IommiClass(
        qualname="iommi.table.Column", bases=[],
        refinables={"after": _r("after", "evaluated_scalar")},
    )
    table = IommiClass(
        qualname="iommi.table.Table", bases=[],
        refinables={
            "columns": _r("columns", "members", member_class="iommi.table.Column"),
            "auto": _r("auto", "open_namespace"),
        },
    )
    graph = IommiGraph(
        iommi_version="0.0-test",
        classes={c.qualname: c for c in [table, column]},
    )
    save_graph(graph, tmp_path / GRAPH_FILENAME)
    a = IommiAnalyzer(workspace_root=tmp_path)
    asyncio.run(a.index(tmp_path))

    uri, pos = _write_with_cursor(
        tmp_path, "from iommi import Table\nTable(auto__mo"
    )
    result = a.completions(uri, pos)
    assert result.exclusive is True
    assert _labels(result) == ["auto__model"]


def test_auto_namespace_keys(analyzer, tmp_path):
    # `Table(auto__` → `model`, `rows`, `instance`, `include`, `exclude`.
    uri, pos = _write_with_cursor(
        tmp_path, "from iommi import Table\nTable(auto__"
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    labels = set(_labels(result))
    assert "auto__model" in labels
    assert "auto__include" in labels
    assert "auto__exclude" in labels
    # No type info for namespace keys → default to `=`.
    for it in result.items:
        assert it["insertText"].endswith("=")


# ---------------------------------------------------------------------------
# auto__model → field-name completions in the members slot
# ---------------------------------------------------------------------------


def test_columns_after_auto_model_still_drills(analyzer_with_django, tmp_path):
    # `Table(auto__model=User, col` → `columns__`.
    uri, pos = _write_with_cursor(
        tmp_path,
        "from iommi import Table\nfrom myapp.models import User\n"
        "Table(auto__model=User, col",
    )
    result = analyzer_with_django.completions(uri, pos)
    assert result.exclusive is True
    assert _labels(result) == ["columns__"]


def test_columns_member_slot_offers_model_fields(analyzer_with_django, tmp_path):
    # `Table(auto__model=User, columns__` → User's fields.
    uri, pos = _write_with_cursor(
        tmp_path,
        "from iommi import Table\nfrom myapp.models import User\n"
        "Table(auto__model=User, columns__",
    )
    result = analyzer_with_django.completions(uri, pos)
    assert result.exclusive is True
    labels = set(_labels(result))
    # Each entry drills in (the user typically configures attrs on an
    # auto-generated column).
    assert "columns__username__" in labels
    assert "columns__email__" in labels


def test_columns_member_slot_with_partial(analyzer_with_django, tmp_path):
    uri, pos = _write_with_cursor(
        tmp_path,
        "from iommi import Table\nfrom myapp.models import User\n"
        "Table(auto__model=User, columns__us",
    )
    result = analyzer_with_django.completions(uri, pos)
    assert result.exclusive is True
    assert _labels(result) == ["columns__username__"]


def test_fields_member_slot_via_form_class_ref(analyzer_with_django, tmp_path):
    # Form has a `fields` members refinable. Through Table.bulk (class_ref
    # → Form) plus an auto__model, the same trick applies.
    uri, pos = _write_with_cursor(
        tmp_path,
        "from iommi import Form\nfrom myapp.models import User\n"
        "Form(auto__model=User, fields__",
    )
    result = analyzer_with_django.completions(uri, pos)
    assert result.exclusive is True
    labels = set(_labels(result))
    assert "fields__username__" in labels


def test_member_slot_silent_without_auto_model(analyzer_with_django, tmp_path):
    # No `auto__model` in the call → we don't invent fields.
    uri, pos = _write_with_cursor(
        tmp_path, "from iommi import Table\nTable(columns__"
    )
    result = analyzer_with_django.completions(uri, pos)
    assert result.exclusive is True
    assert result.items == []


def test_auto_rows_manager_chain_resolves_model(analyzer_with_django, tmp_path):
    # `auto__rows=User.objects.all()` should also bind the model.
    uri, pos = _write_with_cursor(
        tmp_path,
        "from iommi import Table\nfrom myapp.models import User\n"
        "Table(auto__rows=User.objects.all(), columns__",
    )
    result = analyzer_with_django.completions(uri, pos)
    labels = set(_labels(result))
    assert "columns__username__" in labels


def test_auto_instance_manager_chain_resolves_model(analyzer_with_django, tmp_path):
    uri, pos = _write_with_cursor(
        tmp_path,
        "from iommi import Table\nfrom myapp.models import User\n"
        "Table(auto__instance=User.objects.get(pk=1), columns__",
    )
    result = analyzer_with_django.completions(uri, pos)
    labels = set(_labels(result))
    assert "columns__username__" in labels


def test_auto_model_without_django_index_silent(analyzer, tmp_path):
    # The analyzer without a django_index_provider must not invent fields.
    uri, pos = _write_with_cursor(
        tmp_path,
        "from iommi import Table\nTable(auto__model=User, columns__",
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    assert result.items == []


# ---------------------------------------------------------------------------
# auto__include / auto__exclude: string-literal field completion
# ---------------------------------------------------------------------------


def test_auto_include_string_literal(analyzer_with_django, tmp_path):
    uri, pos = _write_with_cursor(
        tmp_path,
        "from iommi import Table\nfrom myapp.models import User\n"
        "Table(auto__model=User, auto__include=['us",
    )
    result = analyzer_with_django.completions(uri, pos)
    assert result.exclusive is True
    assert _labels(result) == ["username"]
    # Bare field name — no `=` or `__` because the user is inside a string.
    assert result.items[0]["insertText"] == "username"


def test_auto_exclude_string_literal_double_quote(analyzer_with_django, tmp_path):
    uri, pos = _write_with_cursor(
        tmp_path,
        'from iommi import Table\nfrom myapp.models import User\n'
        'Table(auto__model=User, auto__exclude=["em',
    )
    result = analyzer_with_django.completions(uri, pos)
    assert result.exclusive is True
    assert _labels(result) == ["email"]


def test_auto_include_string_literal_empty_partial(analyzer_with_django, tmp_path):
    uri, pos = _write_with_cursor(
        tmp_path,
        "from iommi import Table\nfrom myapp.models import User\n"
        "Table(auto__model=User, auto__include=['",
    )
    result = analyzer_with_django.completions(uri, pos)
    assert result.exclusive is True
    labels = set(_labels(result))
    assert "username" in labels
    assert "email" in labels


def test_auto_include_string_literal_with_existing_entries(
    analyzer_with_django, tmp_path,
):
    uri, pos = _write_with_cursor(
        tmp_path,
        "from iommi import Table\nfrom myapp.models import User\n"
        "Table(auto__model=User, auto__include=['username', 'em",
    )
    result = analyzer_with_django.completions(uri, pos)
    assert result.exclusive is True
    assert _labels(result) == ["email"]


def test_auto_include_without_auto_model_silent(analyzer_with_django, tmp_path):
    # No `auto__model` → we can't pick a model, so non-exclusive empty.
    uri, pos = _write_with_cursor(
        tmp_path,
        "from iommi import Table\nTable(auto__include=['us",
    )
    result = analyzer_with_django.completions(uri, pos)
    assert result.exclusive is False
    assert result.items == []


def test_string_in_unrelated_kwarg_is_silent(analyzer_with_django, tmp_path):
    # A string in some other kwarg shouldn't trigger us — let ty handle.
    uri, pos = _write_with_cursor(
        tmp_path,
        "from iommi import Table\nfrom myapp.models import User\n"
        "Table(auto__model=User, page_size='hu",
    )
    result = analyzer_with_django.completions(uri, pos)
    assert result.exclusive is False
    assert result.items == []
