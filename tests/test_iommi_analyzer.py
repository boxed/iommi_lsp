"""Integration tests for IommiAnalyzer over a workspace + graph fixture."""

from __future__ import annotations

import asyncio
import textwrap
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
                sub_specials={"class": {"value_type": "bool"}, "style": {"value_type": "str"}},
            ),
        },
    )
    table = IommiClass(
        qualname="iommi.table.Table",
        bases=["iommi.part.Part"],
        refinables={
            "columns": _r("columns", "members", member_class="iommi.table.Column"),
            "page_size": _r("page_size", "evaluated_scalar"),
            "attrs": _r(
                "attrs", "html_attrs",
                sub_specials={"class": {"value_type": "bool"}, "style": {"value_type": "str"}},
            ),
        },
    )
    return IommiGraph(
        iommi_version="0.0-test",
        classes={c.qualname: c for c in [table, column, part]},
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    save_graph(_make_fixture_graph(), tmp_path / GRAPH_FILENAME)
    return tmp_path


def _write_usage(workspace: Path, src: str) -> Path:
    f = workspace / "usage.py"
    f.write_text(textwrap.dedent(src).lstrip())
    return f


def _diagnose(workspace: Path, source: str) -> list[dict]:
    f = _write_usage(workspace, source)
    a = IommiAnalyzer(workspace_root=workspace)
    asyncio.run(a.index(workspace))
    return a.additional_diagnostics(f.as_uri())


def test_no_graph_means_no_diagnostics(tmp_path: Path):
    f = tmp_path / "usage.py"
    f.write_text("from iommi import Table\nTable(bogus=1)\n")
    # Disable auto-build so the test really exercises "no graph" — in CI
    # iommi is importable so an in-process build would otherwise succeed.
    a = IommiAnalyzer(workspace_root=tmp_path, auto_build=False)
    asyncio.run(a.index(tmp_path))
    assert a.additional_diagnostics(f.as_uri()) == []


def test_valid_call_produces_no_diagnostics(workspace: Path):
    diags = _diagnose(workspace, """
        from iommi import Table

        t = Table(columns__name__after="x", page_size=10)
    """)
    assert diags == []


def test_unknown_top_level_kwarg(workspace: Path):
    diags = _diagnose(workspace, """
        from iommi import Table

        t = Table(bogus_thing=1)
    """)
    assert len(diags) == 1
    d = diags[0]
    assert d["code"] == "iommi-unknown-refinable"
    assert d["source"] == "iommi_lsp"
    assert "bogus_thing" in d["message"]
    # Range pinned to the kwarg name.
    src = (workspace / "usage.py").read_text()
    line = src.splitlines()[d["range"]["start"]["line"]]
    col = d["range"]["start"]["character"]
    assert line[col:col + len("bogus_thing")] == "bogus_thing"


def test_unknown_member_refinable_pins_to_bad_segment(workspace: Path):
    diags = _diagnose(workspace, """
        from iommi import Table

        t = Table(columns__name__bogus_thing=1)
    """)
    assert len(diags) == 1
    d = diags[0]
    src = (workspace / "usage.py").read_text()
    line = src.splitlines()[d["range"]["start"]["line"]]
    col = d["range"]["start"]["character"]
    assert line[col:col + len("bogus_thing")] == "bogus_thing"
    # Segment offset within the kwarg, not the full name's start.
    assert "columns" not in line[col:col + 10]


def test_chain_past_scalar(workspace: Path):
    diags = _diagnose(workspace, """
        from iommi import Table

        t = Table(page_size__bogus=1)
    """)
    assert len(diags) == 1
    assert diags[0]["data"]["outcome"] == "trailing_segments_after_leaf"


def test_html_attrs_direct_attribute_ok(workspace: Path):
    diags = _diagnose(workspace, """
        from iommi import Table

        t = Table(attrs__data_thing="hi")
    """)
    assert diags == []


def test_html_attrs_class_subspecial_ok(workspace: Path):
    diags = _diagnose(workspace, """
        from iommi import Table

        t = Table(attrs__class__bold=True)
    """)
    assert diags == []


def test_html_attrs_chain_past_class_value_flagged(workspace: Path):
    diags = _diagnose(workspace, """
        from iommi import Table

        t = Table(attrs__class__bold__nope=True)
    """)
    assert len(diags) == 1
    assert diags[0]["data"]["outcome"] == "trailing_segments_after_leaf"


def test_attrs_directly_on_column_passes_via_heuristic(workspace: Path):
    """`attrs` is the universal iommi escape hatch — any segment named
    ``attrs`` reached from an iommi class behaves like html_attrs even
    when the static reflector missed it (custom Tag mixins, etc.)."""
    diags = _diagnose(workspace, """
        from iommi import Table

        t = Table(columns__name__attrs__class__bold=True)
    """)
    assert diags == []


def test_attrs_inside_cell_namespace_recurses_into_html_attrs(workspace: Path):
    """The "right" path the user originally pointed at: cell.attrs.class.x"""
    diags = _diagnose(workspace, """
        from iommi import Table

        t = Table(columns__name__cell__attrs__class__bold=True)
    """)
    assert diags == []


def test_attrs_inside_cell_chain_past_class_value_fails(workspace: Path):
    diags = _diagnose(workspace, """
        from iommi import Table

        t = Table(columns__name__cell__attrs__class__bold__deeper=True)
    """)
    assert len(diags) == 1
    assert diags[0]["data"]["outcome"] == "trailing_segments_after_leaf"


def test_unknown_class_silently_passes(workspace: Path):
    diags = _diagnose(workspace, """
        class NotIommi:
            def __init__(self, **kw): pass

        x = NotIommi(bogus_thing=1)
    """)
    assert diags == []


def test_kwargs_splat_is_skipped(workspace: Path):
    diags = _diagnose(workspace, """
        from iommi import Table
        opts = {"bogus": 1}
        t = Table(**opts)
    """)
    assert diags == []


def test_module_qualified_class_resolves(workspace: Path):
    diags = _diagnose(workspace, """
        import iommi
        t = iommi.Table(bogus_thing=1)
    """)
    assert len(diags) == 1
    assert "bogus_thing" in diags[0]["message"]


def _make_traditional_cell_graph() -> IommiGraph:
    """Graph where Column.cell is a traditional_class targeting Cell —
    i.e. its chain segments validate against Cell.__init__'s public
    self-assignments rather than Meta-derived namespace keys."""
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
        },
    )
    table = IommiClass(
        qualname="iommi.table.Table",
        bases=[],
        refinables={
            "columns": _r(
                "columns", "members", member_class="iommi.table.Column",
            ),
            "auto": _r(
                "auto", "namespace",
                known_keys=["model", "rows", "instance", "include", "exclude"],
            ),
        },
    )
    return IommiGraph(
        iommi_version="0.0-test",
        classes={c.qualname: c for c in [table, column, cell]},
    )


@pytest.fixture
def traditional_workspace(tmp_path: Path) -> Path:
    save_graph(_make_traditional_cell_graph(), tmp_path / GRAPH_FILENAME)
    return tmp_path


def test_cell_value_init_member_is_valid(traditional_workspace: Path):
    """``columns__email__cell__value`` chains through a traditional_class
    into Cell's init members — ``value`` is one, so no diagnostic."""
    diags = _diagnose(traditional_workspace, """
        from iommi import Table

        t = Table(
            auto__model=User,
            columns__email__cell__value=lambda row, **_: row.email,
        )
    """)
    assert diags == []


def test_cell_value_on_direct_column_call_is_valid(traditional_workspace: Path):
    """The user-reported case: ``Column(cell__value=…, cell__url=…)``
    in declarative-table style. ``value`` and ``url`` are both Cell
    init members, so neither should be flagged."""
    diags = _diagnose(traditional_workspace, """
        from iommi import Column, Table

        t = Table(
            columns=dict(
                project=Column(
                    cell__value=lambda row, **_: row.get_short_name(),
                    cell__url=lambda row, **_: row.get_absolute_url(),
                ),
            ),
        )
    """)
    assert diags == []


def test_cell_unknown_init_member_is_flagged(traditional_workspace: Path):
    """``val`` is not a member of Cell — analyzer should flag it and pin
    the diagnostic to the bad segment within the kwarg name."""
    diags = _diagnose(traditional_workspace, """
        from iommi import Table

        t = Table(
            auto__model=User,
            columns__email__cell__val=lambda row, **_: row.email,
        )
    """)
    assert len(diags) == 1
    d = diags[0]
    assert d["code"] == "iommi-unknown-refinable"
    assert d["data"]["outcome"] == "unknown_refinable"
    assert d["data"]["on_class"] == "iommi.table.Cell"
    assert "val" in d["message"]
    assert "value" in d["data"]["available"]
    # The range pins to `val`, not the full kwarg name.
    src = (traditional_workspace / "usage.py").read_text()
    line = src.splitlines()[d["range"]["start"]["line"]]
    col_start = d["range"]["start"]["character"]
    col_end = d["range"]["end"]["character"]
    assert line[col_start:col_end] == "val"


# ---------------------------------------------------------------------------
# class Meta pattern (semantically equivalent to kwargs passed to the base)
# ---------------------------------------------------------------------------


def test_class_meta_valid_kwargs(workspace: Path):
    diags = _diagnose(workspace, """
        from iommi import Table

        class MyTable(Table):
            class Meta:
                page_size = 10
                columns__name__after = "x"
    """)
    assert diags == []


def test_class_meta_unknown_top_level_kwarg(workspace: Path):
    diags = _diagnose(workspace, """
        from iommi import Table

        class MyTable(Table):
            class Meta:
                bogus_thing = 1
    """)
    assert len(diags) == 1
    d = diags[0]
    assert d["code"] == "iommi-unknown-refinable"
    assert d["source"] == "iommi_lsp"
    assert "bogus_thing" in d["message"]
    # Range pinned to the attribute name in the Meta body.
    src = (workspace / "usage.py").read_text()
    line = src.splitlines()[d["range"]["start"]["line"]]
    col_start = d["range"]["start"]["character"]
    col_end = d["range"]["end"]["character"]
    assert line[col_start:col_end] == "bogus_thing"


def test_class_meta_unknown_chain_segment_pins_to_bad_segment(workspace: Path):
    diags = _diagnose(workspace, """
        from iommi import Table

        class MyTable(Table):
            class Meta:
                columns__name__bogus_thing = 1
    """)
    assert len(diags) == 1
    d = diags[0]
    src = (workspace / "usage.py").read_text()
    line = src.splitlines()[d["range"]["start"]["line"]]
    col_start = d["range"]["start"]["character"]
    col_end = d["range"]["end"]["character"]
    assert line[col_start:col_end] == "bogus_thing"


def test_class_meta_chain_past_scalar(workspace: Path):
    diags = _diagnose(workspace, """
        from iommi import Table

        class MyTable(Table):
            class Meta:
                page_size__nope = 1
    """)
    assert len(diags) == 1
    assert diags[0]["data"]["outcome"] == "trailing_segments_after_leaf"


def test_class_meta_html_attrs_ok(workspace: Path):
    diags = _diagnose(workspace, """
        from iommi import Table

        class MyTable(Table):
            class Meta:
                attrs__class__bold = True
                attrs__data_thing = "hi"
    """)
    assert diags == []


def test_class_meta_ann_assign_treated_as_kwarg(workspace: Path):
    """Annotated assignments inside Meta count too — `page_size: int = 10`
    is just as valid as `page_size = 10`."""
    diags = _diagnose(workspace, """
        from iommi import Table

        class MyTable(Table):
            class Meta:
                page_size: int = 10
                bogus_thing: int = 1
    """)
    assert len(diags) == 1
    assert "bogus_thing" in diags[0]["message"]


def test_class_meta_on_non_iommi_class_is_silent(workspace: Path):
    diags = _diagnose(workspace, """
        class Plain:
            class Meta:
                bogus = 1
    """)
    assert diags == []


def test_class_meta_on_user_iommi_subclass_is_flagged(workspace: Path):
    """A subclass of a known iommi class is itself iommi for Meta validation.
    `class MyOtherTable(MyTable)` should validate against Table's refinables."""
    diags = _diagnose(workspace, """
        from iommi import Table

        class MyTable(Table):
            pass

        class MyOtherTable(MyTable):
            class Meta:
                bogus_thing = 1
    """)
    assert len(diags) == 1
    assert "bogus_thing" in diags[0]["message"]


def test_class_body_assignment_outside_meta_is_silent(workspace: Path):
    """Outside `class Meta:` the assignments are declarative members
    (`name = Column()`) or methods — the names are user-defined, not
    refinables, so we must not flag them."""
    diags = _diagnose(workspace, """
        from iommi import Table

        class MyTable(Table):
            name = "not a refinable check target"
            bogus_thing = 1

            def helper(self):
                pass
    """)
    assert diags == []


def test_class_meta_module_qualified_base(workspace: Path):
    diags = _diagnose(workspace, """
        import iommi

        class MyTable(iommi.Table):
            class Meta:
                bogus_thing = 1
    """)
    assert len(diags) == 1
    assert "bogus_thing" in diags[0]["message"]


def test_class_meta_traditional_cell_chain(traditional_workspace: Path):
    """class Meta entries should walk into traditional_class refinables
    (e.g. ``Column.cell``) just like kwargs do."""
    diags = _diagnose(traditional_workspace, """
        from iommi import Table

        class MyTable(Table):
            class Meta:
                auto__model = User
                columns__email__cell__value = lambda row, **_: row.email
                columns__email__cell__val = lambda row, **_: row.email
    """)
    assert len(diags) == 1
    d = diags[0]
    assert d["data"]["on_class"] == "iommi.table.Cell"
    assert "val" in d["message"]


# ---------------------------------------------------------------------------
# Table(header__...) / Column(header__...). ``Table.header`` is declared
# as a bare ``Refinable()`` but at runtime ``on_refine_done`` unpacks it
# into a ``HeaderConfig``; static reflection would otherwise classify it
# as a scalar leaf and reject every valid ``header__template=``-style
# usage. ``Column.header: Namespace = EvaluatedRefinable()`` is the same
# story for ``HeaderColumnConfig``.
# ---------------------------------------------------------------------------


def _make_header_config_graph() -> IommiGraph:
    """Graph that mirrors the real-iommi shape after the class_ref
    override is applied: ``Table.header`` / ``Column.header`` /
    ``Table.superheader`` all class_ref into HeaderConfig/HeaderColumnConfig."""
    header_config = IommiClass(
        qualname="iommi.table.HeaderConfig",
        bases=[],
        refinables={
            "tag": _r("tag", "evaluated_scalar"),
            "template": _r("template", "evaluated_scalar"),
            "include": _r("include", "evaluated_scalar"),
            "extra": _r("extra", "open_namespace"),
            "extra_evaluated": _r("extra_evaluated", "open_namespace"),
            "attrs": _r(
                "attrs", "html_attrs",
                sub_specials={
                    "class": {"value_type": "bool"},
                    "style": {"value_type": "str"},
                },
            ),
        },
    )
    header_column_config = IommiClass(
        qualname="iommi.table.HeaderColumnConfig",
        bases=[],
        refinables={
            "template": _r("template", "evaluated_scalar"),
            "url": _r("url", "evaluated_scalar"),
            "attrs": _r(
                "attrs", "html_attrs",
                sub_specials={
                    "class": {"value_type": "bool"},
                    "style": {"value_type": "str"},
                },
            ),
        },
    )
    column = IommiClass(
        qualname="iommi.table.Column",
        bases=[],
        refinables={
            "header": _r(
                "header", "class_ref", target="iommi.table.HeaderColumnConfig",
            ),
        },
    )
    table = IommiClass(
        qualname="iommi.table.Table",
        bases=[],
        refinables={
            "columns": _r(
                "columns", "members", member_class="iommi.table.Column",
            ),
            "header": _r("header", "class_ref", target="iommi.table.HeaderConfig"),
            "superheader": _r(
                "superheader", "class_ref", target="iommi.table.HeaderConfig",
            ),
        },
    )
    return IommiGraph(
        iommi_version="0.0-test",
        classes={c.qualname: c for c in [
            table, column, header_config, header_column_config,
        ]},
    )


@pytest.fixture
def header_workspace(tmp_path: Path) -> Path:
    save_graph(_make_header_config_graph(), tmp_path / GRAPH_FILENAME)
    return tmp_path


def test_table_header_template_ok(header_workspace: Path):
    """``Table(header__template=...)`` is the user-reported regression —
    must validate cleanly."""
    diags = _diagnose(header_workspace, """
        from iommi import Table

        Table(header__template='custom.html')
    """)
    assert diags == []


def test_table_header_tag_ok(header_workspace: Path):
    diags = _diagnose(header_workspace, """
        from iommi import Table

        Table(header__tag='div')
    """)
    assert diags == []


def test_table_header_include_ok(header_workspace: Path):
    diags = _diagnose(header_workspace, """
        from iommi import Table

        Table(header__include=False)
    """)
    assert diags == []


def test_table_header_extra_ok(header_workspace: Path):
    """``extra``/``extra_evaluated`` are open buckets — any sub-key OK."""
    diags = _diagnose(header_workspace, """
        from iommi import Table

        Table(
            header__extra__custom_key=1,
            header__extra_evaluated__custom=lambda **_: 2,
        )
    """)
    assert diags == []


def test_table_header_attrs_class_ok(header_workspace: Path):
    diags = _diagnose(header_workspace, """
        from iommi import Table

        Table(header__attrs__class__bold=True)
    """)
    assert diags == []


def test_table_header_attrs_style_ok(header_workspace: Path):
    diags = _diagnose(header_workspace, """
        from iommi import Table

        Table(header__attrs__style__color='red')
    """)
    assert diags == []


def test_table_header_attrs_direct_attribute_ok(header_workspace: Path):
    diags = _diagnose(header_workspace, """
        from iommi import Table

        Table(header__attrs__data_section='hi')
    """)
    assert diags == []


def test_table_header_unknown_refinable_flagged(header_workspace: Path):
    """Unknown sub-refinable on HeaderConfig surfaces the available list."""
    diags = _diagnose(header_workspace, """
        from iommi import Table

        Table(header__bogus_thing=1)
    """)
    assert len(diags) == 1
    d = diags[0]
    assert d["data"]["outcome"] == "unknown_refinable"
    assert d["data"]["on_class"] == "iommi.table.HeaderConfig"
    assert "bogus_thing" in d["message"]
    assert "template" in d["data"]["available"]


def test_table_header_chain_past_template_flagged(header_workspace: Path):
    """``header__template`` is a scalar leaf — nothing past it."""
    src_template = """
        from iommi import Table

        Table(header__template__nope=1)
    """
    diags = _diagnose(header_workspace, src_template)
    assert len(diags) == 1
    d = diags[0]
    assert d["data"]["outcome"] == "trailing_segments_after_leaf"
    assert "nope" in d["message"]
    # The diagnostic pins to the bad segment within the kwarg.
    src = (header_workspace / "usage.py").read_text()
    line = src.splitlines()[d["range"]["start"]["line"]]
    col_start = d["range"]["start"]["character"]
    col_end = d["range"]["end"]["character"]
    assert line[col_start:col_end] == "nope"


def test_table_superheader_template_ok(header_workspace: Path):
    """``Table.superheader`` is configured via ``with_defaults`` at runtime
    with ``superheader__template=...`` and ``superheader__attrs__class__...``
    — the override class_refs it to HeaderConfig so those refinements pass."""
    diags = _diagnose(header_workspace, """
        from iommi import Table

        Table(
            superheader__template='x.html',
            superheader__attrs__class__superheader=True,
        )
    """)
    assert diags == []


def test_column_header_template_ok(header_workspace: Path):
    diags = _diagnose(header_workspace, """
        from iommi import Column

        Column(header__template='col-header.html')
    """)
    assert diags == []


def test_column_header_url_ok(header_workspace: Path):
    """``HeaderColumnConfig.url`` is a refinable not present on the table
    header — make sure the right config class is consulted per ``header``."""
    diags = _diagnose(header_workspace, """
        from iommi import Column

        Column(header__url='/sort')
    """)
    assert diags == []


def test_column_header_attrs_ok(header_workspace: Path):
    diags = _diagnose(header_workspace, """
        from iommi import Column

        Column(header__attrs__class__numeric=True)
    """)
    assert diags == []


def test_column_header_unknown_refinable_flagged(header_workspace: Path):
    """``Column.header`` walks into HeaderColumnConfig, not HeaderConfig —
    a HeaderConfig-only key like ``tag`` should be flagged here."""
    diags = _diagnose(header_workspace, """
        from iommi import Column

        Column(header__tag='div')
    """)
    assert len(diags) == 1
    d = diags[0]
    assert d["data"]["outcome"] == "unknown_refinable"
    assert d["data"]["on_class"] == "iommi.table.HeaderColumnConfig"
    assert "tag" in d["message"]


def test_column_header_inside_columns_chain_ok(header_workspace: Path):
    """The chain ``columns__name__header__template`` works through members."""
    diags = _diagnose(header_workspace, """
        from iommi import Table

        Table(columns__name__header__template='c.html')
    """)
    assert diags == []


def test_table_class_meta_header_template_ok(header_workspace: Path):
    """The same chain through ``class Meta`` — declarative configuration."""
    diags = _diagnose(header_workspace, """
        from iommi import Table

        class MyTable(Table):
            class Meta:
                header__template = 'custom.html'
                header__attrs__class__sticky = True
                superheader__template = 'super.html'
    """)
    assert diags == []


def test_class_meta_header_unknown_refinable_flagged(header_workspace: Path):
    diags = _diagnose(header_workspace, """
        from iommi import Table

        class MyTable(Table):
            class Meta:
                header__bogus_thing = 1
    """)
    assert len(diags) == 1
    assert diags[0]["data"]["on_class"] == "iommi.table.HeaderConfig"
    assert "bogus_thing" in diags[0]["message"]


def test_class_meta_kwargs_and_call_kwargs_coexist(workspace: Path):
    """Meta validation must not interfere with the existing kwarg
    diagnostics on the same file."""
    diags = _diagnose(workspace, """
        from iommi import Table

        class MyTable(Table):
            class Meta:
                bogus_meta = 1

        t = Table(bogus_call=1)
    """)
    messages = sorted(d["message"] for d in diags)
    assert len(messages) == 2
    assert any("bogus_meta" in m for m in messages)
    assert any("bogus_call" in m for m in messages)


def test_class_meta_without_meta_inner_class_is_fine(workspace: Path):
    diags = _diagnose(workspace, """
        from iommi import Table

        class MyTable(Table):
            pass
    """)
    assert diags == []


def test_class_meta_non_name_target_is_skipped(workspace: Path):
    """Tuple/attribute targets inside Meta aren't valid refinable names —
    don't crash on them, just skip."""
    diags = _diagnose(workspace, """
        from iommi import Table

        class MyTable(Table):
            class Meta:
                a = b = 10  # multiple targets — first is a Name, that's fine
    """)
    # `a` and `b` are both unknown refinables on Table.
    msgs = sorted(d["message"] for d in diags)
    assert any("'a'" in m for m in msgs)
    assert any("'b'" in m for m in msgs)


def test_caches_parsed_file(workspace: Path):
    f = _write_usage(workspace, "from iommi import Table\nTable(bogus=1)\n")
    a = IommiAnalyzer(workspace_root=workspace)
    asyncio.run(a.index(workspace))
    uri = f.as_uri()
    a.additional_diagnostics(uri)
    assert uri in a._cache
    asyncio.run(a.on_file_changed(uri))
    assert uri not in a._cache


# ---------------------------------------------------------------------------
# attr= bridging — Form(fields__name__attr='model__path') / Table(columns__c__attr=…)
# ---------------------------------------------------------------------------


def _make_attr_bridge_graph() -> IommiGraph:
    """Graph that exposes ``attr`` as a refinable on Column/Field."""
    column = IommiClass(
        qualname="iommi.table.Column",
        bases=["iommi.part.Part"],
        refinables={
            "attr": _r("attr", "evaluated_scalar"),
        },
    )
    field = IommiClass(
        qualname="iommi.form.Field",
        bases=["iommi.part.Part"],
        refinables={
            "attr": _r("attr", "evaluated_scalar"),
        },
    )
    part = IommiClass(
        qualname="iommi.part.Part",
        bases=[],
        refinables={},
    )
    table = IommiClass(
        qualname="iommi.table.Table",
        bases=["iommi.part.Part"],
        refinables={
            "columns": _r("columns", "members", member_class="iommi.table.Column"),
        },
    )
    form = IommiClass(
        qualname="iommi.form.Form",
        bases=["iommi.part.Part"],
        refinables={
            "fields": _r("fields", "members", member_class="iommi.form.Field"),
        },
    )
    return IommiGraph(
        iommi_version="0.0-test",
        classes={c.qualname: c for c in [table, form, column, field, part]},
    )


@pytest.fixture
def attr_workspace(tmp_path: Path) -> Path:
    save_graph(_make_attr_bridge_graph(), tmp_path / GRAPH_FILENAME)
    return tmp_path


def _attr_analyzer(workspace: Path) -> IommiAnalyzer:
    from iommi_lsp.analyzers.django.index import (
        DjangoIndex,
        FieldInfo,
        ModelInfo,
    )

    user = ModelInfo(
        qualname="myapp.models.User",
        module="myapp.models",
        name="User",
        file_path=workspace / "myapp" / "models.py",
        bases=["django.db.models.Model"],
        fields={
            "username": FieldInfo(name="username", field_type="CharField"),
            "email": FieldInfo(name="email", field_type="EmailField"),
        },
    )
    index = DjangoIndex()
    index.add_model(user)
    a = IommiAnalyzer(
        workspace_root=workspace,
        django_index_provider=lambda: index,
    )
    asyncio.run(a.index(workspace))
    return a


def test_attr_path_validates_against_auto_model(attr_workspace: Path):
    f = _write_usage(attr_workspace, """
        from iommi import Form

        Form(
            auto__model=User,
            fields__nickname__attr='usernme',
        )
    """)
    a = _attr_analyzer(attr_workspace)
    diags = a.additional_diagnostics(f.as_uri())
    assert any(
        d.get("code") == "iommi-unknown-attr-path"
        and "usernme" in d.get("message", "")
        for d in diags
    )


def test_attr_path_valid_value_silent(attr_workspace: Path):
    f = _write_usage(attr_workspace, """
        from iommi import Form

        Form(
            auto__model=User,
            fields__nickname__attr='username',
        )
    """)
    a = _attr_analyzer(attr_workspace)
    diags = a.additional_diagnostics(f.as_uri())
    assert [d for d in diags if d.get("code") == "iommi-unknown-attr-path"] == []


def test_attr_path_table_via_rows(attr_workspace: Path):
    """``rows=Model.objects.all()`` should also bind the model."""
    f = _write_usage(attr_workspace, """
        from iommi import Table

        Table(
            rows=User.objects.all(),
            columns__nickname__attr='nope_field',
        )
    """)
    a = _attr_analyzer(attr_workspace)
    diags = a.additional_diagnostics(f.as_uri())
    assert any(
        d.get("code") == "iommi-unknown-attr-path"
        and "nope_field" in d.get("message", "")
        for d in diags
    )


def test_attr_path_silent_without_auto_model(attr_workspace: Path):
    """No bound model — we can't know what attr resolves against, so silent."""
    f = _write_usage(attr_workspace, """
        from iommi import Form

        Form(fields__nickname__attr='whatever')
    """)
    a = _attr_analyzer(attr_workspace)
    diags = a.additional_diagnostics(f.as_uri())
    assert [d for d in diags if d.get("code") == "iommi-unknown-attr-path"] == []


# ---------------------------------------------------------------------------
# Action(post_handler=…) / endpoints__name__func — callable-expected check
# ---------------------------------------------------------------------------


def _make_callable_leaf_graph() -> IommiGraph:
    """Graph with ``post_handler`` / ``func`` refinables on Action."""
    action = IommiClass(
        qualname="iommi.action.Action",
        bases=[],
        refinables={
            "post_handler": _r("post_handler", "scalar"),
            "tag": _r("tag", "evaluated_scalar"),
        },
    )
    endpoint = IommiClass(
        qualname="iommi.endpoint.Endpoint",
        bases=[],
        refinables={
            "func": _r("func", "scalar"),
        },
    )
    form = IommiClass(
        qualname="iommi.form.Form",
        bases=[],
        refinables={
            "actions": _r(
                "actions", "members", member_class="iommi.action.Action",
            ),
            "endpoints": _r(
                "endpoints", "members", member_class="iommi.endpoint.Endpoint",
            ),
        },
    )
    return IommiGraph(
        iommi_version="0.0-test",
        classes={c.qualname: c for c in [form, action, endpoint]},
    )


@pytest.fixture
def callable_leaf_workspace(tmp_path: Path) -> Path:
    save_graph(_make_callable_leaf_graph(), tmp_path / GRAPH_FILENAME)
    return tmp_path


def test_post_handler_string_flagged(callable_leaf_workspace: Path):
    diags = _diagnose(callable_leaf_workspace, """
        from iommi import Action

        Action(post_handler='save_widget')
    """)
    assert any(
        d.get("code") == "iommi-callable-expected"
        and "'post_handler'" in d.get("message", "")
        for d in diags
    )


def test_post_handler_name_silent(callable_leaf_workspace: Path):
    """Name references go through ty — no iommi-callable-expected."""
    diags = _diagnose(callable_leaf_workspace, """
        from iommi import Action

        def save_widget(form, **_): pass

        Action(post_handler=save_widget)
    """)
    assert [d for d in diags if d.get("code") == "iommi-callable-expected"] == []


def test_post_handler_lambda_silent(callable_leaf_workspace: Path):
    diags = _diagnose(callable_leaf_workspace, """
        from iommi import Action

        Action(post_handler=lambda form, **_: None)
    """)
    assert [d for d in diags if d.get("code") == "iommi-callable-expected"] == []


def test_form_actions_post_handler_chain_flagged(callable_leaf_workspace: Path):
    """``Form(actions__save__post_handler='...')`` — chain ends in
    ``post_handler`` and value is a string literal."""
    diags = _diagnose(callable_leaf_workspace, """
        from iommi import Form

        Form(actions__save__post_handler='go')
    """)
    assert any(
        d.get("code") == "iommi-callable-expected" for d in diags
    )


def test_endpoints_func_chain_flagged(callable_leaf_workspace: Path):
    diags = _diagnose(callable_leaf_workspace, """
        from iommi import Form

        Form(endpoints__my_endpoint__func='handle')
    """)
    assert any(
        d.get("code") == "iommi-callable-expected" for d in diags
    )


def test_attr_path_diagnostic_pins_bad_segment(attr_workspace: Path):
    f = _write_usage(attr_workspace, """
        from iommi import Form

        Form(
            auto__model=User,
            fields__nickname__attr='emailx',
        )
    """)
    a = _attr_analyzer(attr_workspace)
    diags = [
        d for d in a.additional_diagnostics(f.as_uri())
        if d.get("code") == "iommi-unknown-attr-path"
    ]
    assert len(diags) == 1
    d = diags[0]
    src = f.read_text()
    line = src.splitlines()[d["range"]["start"]["line"]]
    col_start = d["range"]["start"]["character"]
    col_end = d["range"]["end"]["character"]
    assert line[col_start:col_end] == "emailx"


# ---------------------------------------------------------------------------
# Background graph refresh on startup
# ---------------------------------------------------------------------------


def test_stale_graph_is_refreshed_in_background(tmp_path: Path):
    """A graph file written by an older iommi_lsp can be missing
    refinables the current reflector knows about (real-world case:
    pre-fix graphs lacked ``@refinable``-decorated methods like
    ``Table.preprocess_row``). ``index()`` should load the stale graph
    for immediate use, then atomically swap in a freshly reflected one.
    """
    pytest.importorskip("iommi")   # in-process rebuild needs iommi here.

    # Stale graph: looks like iommi.table.Table but with an empty set of
    # refinables (so anything will be flagged as unknown until the
    # background rebuild swaps it out).
    stale_table = IommiClass(
        qualname="iommi.table.Table",
        bases=[],
        refinables={},
    )
    save_graph(
        IommiGraph(
            iommi_version="0.0-stale",
            classes={stale_table.qualname: stale_table},
        ),
        tmp_path / GRAPH_FILENAME,
    )

    async def run():
        a = IommiAnalyzer(workspace_root=tmp_path)
        await a.index(tmp_path)
        # Immediately after index(): the loaded (stale) graph is live.
        loaded_table = a.graph.get("iommi.table.Table")
        assert loaded_table is not None
        assert "preprocess_row" not in loaded_table.refinables
        # Wait for the background rebuild to finish.
        assert a._rebuild_task is not None
        await a._rebuild_task
        return a

    a = asyncio.run(run())

    # After the swap the fresh graph is live — preprocess_row, a
    # ``@refinable``-decorated method on Table, is now known.
    table = a.graph.get("iommi.table.Table")
    assert table is not None
    assert "preprocess_row" in table.refinables, (
        "background rebuild did not swap in the fresh reflector graph"
    )


def test_background_refresh_failure_keeps_loaded_graph(
    tmp_path: Path, monkeypatch
):
    """If the background rebuild fails (e.g. iommi can't be imported in
    any candidate venv), we must not clobber the loaded graph."""
    stale_table = IommiClass(
        qualname="iommi.table.Table",
        bases=[],
        refinables={},
    )
    save_graph(
        IommiGraph(
            iommi_version="0.0-stale",
            classes={stale_table.qualname: stale_table},
        ),
        tmp_path / GRAPH_FILENAME,
    )

    from iommi_lsp.analyzers.iommi import analyzer as analyzer_mod
    monkeypatch.setattr(analyzer_mod, "_try_build_graph", lambda _p: None)

    async def run():
        a = IommiAnalyzer(workspace_root=tmp_path)
        await a.index(tmp_path)
        assert a._rebuild_task is not None
        await a._rebuild_task
        return a

    a = asyncio.run(run())

    # Loaded graph unchanged — single class, no preprocess_row.
    assert set(a.graph.classes) == {"iommi.table.Table"}
    assert a.graph.iommi_version == "0.0-stale"


def test_no_background_rebuild_when_auto_build_disabled(tmp_path: Path):
    """``auto_build=False`` opts out of all build paths, including the
    new background refresh. Tests that assert "no graph" or "exactly
    this loaded graph" behaviour rely on this."""
    stale_table = IommiClass(
        qualname="iommi.table.Table",
        bases=[],
        refinables={},
    )
    save_graph(
        IommiGraph(
            iommi_version="0.0-stale",
            classes={stale_table.qualname: stale_table},
        ),
        tmp_path / GRAPH_FILENAME,
    )

    a = IommiAnalyzer(workspace_root=tmp_path, auto_build=False)
    asyncio.run(a.index(tmp_path))
    assert a._rebuild_task is None
    assert a.graph.iommi_version == "0.0-stale"
