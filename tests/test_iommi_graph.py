"""Round-trip + reflector tests for the iommi graph.

Reflector runs against the real iommi (it's a dev dep), so this also
catches breakage when iommi changes its Refinable surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iommi_lsp.analyzers.iommi.graph import (
    GRAPH_FILENAME,
    IommiClass,
    IommiGraph,
    Refinable,
    from_json,
    load_graph,
    save_graph,
    to_json,
)


def test_round_trip_minimal_graph(tmp_path: Path):
    g = IommiGraph(
        iommi_version="7.25.1",
        classes={
            "x.Y": IommiClass(
                qualname="x.Y",
                bases=["x.Base"],
                refinables={
                    "columns": Refinable(
                        name="columns", kind="members",
                        member_class="x.Col", refinable_type="RefinableMembers",
                    ),
                    "attrs": Refinable(
                        name="attrs", kind="html_attrs",
                        refinable_type="SpecialEvaluatedRefinable",
                        sub_specials={"class": {"value_type": "bool"}},
                    ),
                },
            ),
        },
    )
    f = tmp_path / GRAPH_FILENAME
    save_graph(g, f)
    g2 = load_graph(f)
    assert g2 is not None
    assert g2.iommi_version == "7.25.1"
    assert g2.has("x.Y")
    cols = g2.get("x.Y").refinables["columns"]
    assert cols.kind == "members"
    assert cols.member_class == "x.Col"
    attrs = g2.get("x.Y").refinables["attrs"]
    assert attrs.kind == "html_attrs"
    assert attrs.sub_specials == {"class": {"value_type": "bool"}}


def test_load_missing_graph_returns_none(tmp_path: Path):
    assert load_graph(tmp_path / "nope.json") is None


def test_load_corrupt_graph_returns_none(tmp_path: Path):
    f = tmp_path / "broken.json"
    f.write_text("{not json")
    assert load_graph(f) is None


def test_load_rejects_older_schema_version(tmp_path: Path):
    """Older schemas (v1) lack ``traditional_class`` / ``init_members``.
    Loading must return None so the analyzer's index step rebuilds the
    graph — otherwise stale on-disk graphs silently produce wrong
    diagnostics for users who upgrade iommi_lsp without rebuilding."""
    f = tmp_path / "old.json"
    f.write_text(json.dumps({
        "schema_version": 1,
        "iommi_version": "7.0.0",
        "classes": {},
    }))
    assert load_graph(f) is None


def test_lookup_simple_returns_unique_match(tmp_path: Path):
    g = IommiGraph(classes={
        "iommi.table.Table": IommiClass(qualname="iommi.table.Table", bases=[]),
        "iommi.form.Form": IommiClass(qualname="iommi.form.Form", bases=[]),
    })
    assert g.lookup_simple("Table").qualname == "iommi.table.Table"
    assert g.lookup_simple("Nope") is None


# ---------------------------------------------------------------------------
# Reflector tests against real iommi (skipped if iommi isn't installed)
# ---------------------------------------------------------------------------


iommi = pytest.importorskip("iommi")


def test_reflector_classifies_table_correctly():
    from iommi_lsp.analyzers.iommi.reflect import build

    g = build()
    assert g.has("iommi.table.Table")
    table = g.get("iommi.table.Table")

    # columns: Dict[str, Column] = RefinableMembers()
    cols = table.refinables["columns"]
    assert cols.kind == "members"
    assert cols.member_class == "iommi.table.Column"

    # attrs: special with two sub-specials
    attrs = table.refinables["attrs"]
    assert attrs.kind == "html_attrs"
    assert "class" in attrs.sub_specials
    assert "style" in attrs.sub_specials

    # bulk: Optional[Form] = EvaluatedRefinable() — annotation wins over Namespace default
    bulk = table.refinables["bulk"]
    assert bulk.kind == "class_ref"
    assert bulk.target == "iommi.form.Form"


def test_reflector_transitively_includes_targets():
    from iommi_lsp.analyzers.iommi.reflect import build

    g = build()
    # Transitively reachable from Table.bulk -> Form, Table.columns -> Column, etc.
    for q in (
        "iommi.table.Column",
        "iommi.form.Form",
        "iommi.action.Action",
    ):
        assert g.has(q), f"missing {q} from transitive walk"


def test_reflector_records_iommi_version():
    from iommi_lsp.analyzers.iommi.reflect import build

    g = build()
    assert g.iommi_version is not None
    # Loose check — the format is "X.Y.Z" but we don't pin to one version.
    assert g.iommi_version[0].isdigit()


def test_reflector_classifies_cell_as_traditional_class():
    """``Column.cell`` looks like a ``Namespace`` refinable statically, but at
    runtime its kwargs flow into ``Cell.__init__``. The reflector overrides
    that to ``traditional_class`` so completion drills into Cell's init
    members instead of the (incomplete) Meta-derived namespace keys.
    """
    from iommi_lsp.analyzers.iommi.reflect import build

    g = build()
    col = g.get("iommi.table.Column")
    cell_ref = col.refinables["cell"]
    assert cell_ref.kind == "traditional_class"
    assert cell_ref.target == "iommi.table.Cell"

    table = g.get("iommi.table.Table")
    table_cell = table.refinables["cell"]
    assert table_cell.kind == "traditional_class"
    assert table_cell.target == "iommi.table.Cell"

    cell = g.get("iommi.table.Cell")
    assert cell is not None
    # CellConfig.__init__ keyword params land as `self.X = X` assignments;
    # Cell.__init__ adds more. The exact set depends on iommi's version, so
    # spot-check a stable subset.
    members = set(cell.init_members)
    assert {"url", "url_title", "value", "contents", "format", "link"} <= members


def test_reflector_picks_up_refinable_decorated_methods():
    """``@refinable`` (and ``@evaluated_refinable``) on a method declares
    a refinable just like ``Refinable()`` does. The reflector must surface
    those names too, otherwise valid usages like
    ``Table(preprocess_row=lambda row, **_: row)`` get flagged as
    ``unknown-iommi-refinable``.
    """
    from iommi_lsp.analyzers.iommi.reflect import build

    g = build()
    table = g.get("iommi.table.Table")
    assert "preprocess_row" in table.refinables
    assert "preprocess_rows" in table.refinables
    assert "post_bulk_edit" in table.refinables


def test_reflector_classifies_extra_and_extra_evaluated_as_open_namespace():
    """``extra`` and ``extra_evaluated`` are open-ended namespaces in iommi —
    user code can stuff arbitrary keys into them (``Form(extra__my_thing=1)``,
    ``Table(extra_evaluated__color=lambda **_: "red")``). The reflector must
    classify them as ``open_namespace`` everywhere they appear, otherwise
    valid user code gets flagged as ``unknown-iommi-refinable``.
    """
    from iommi_lsp.analyzers.iommi.reflect import build

    g = build()
    seen = 0
    for cls in g.classes.values():
        for name in ("extra", "extra_evaluated"):
            ref = cls.refinables.get(name)
            if ref is None:
                continue
            seen += 1
            assert ref.kind == "open_namespace", (
                f"{cls.qualname}.{name} classified as {ref.kind!r}; "
                "expected 'open_namespace'"
            )
    # Sanity: we actually exercised the assertion against real classes.
    assert seen > 0


def test_reflector_classifies_header_as_class_ref():
    """``Table.header`` is declared as a bare ``Refinable()`` with no
    annotation and no Meta default. Static reflection alone would mark
    it as a scalar leaf and reject ``Table(header__template=...)``.
    The override patches it to ``class_ref`` -> HeaderConfig, and the
    BFS walk pulls HeaderConfig into the graph with its full surface
    (template, tag, include, extra, attrs, ...).
    """
    from iommi_lsp.analyzers.iommi.reflect import build

    g = build()

    table = g.get("iommi.table.Table")
    header = table.refinables["header"]
    assert header.kind == "class_ref"
    assert header.target == "iommi.table.HeaderConfig"

    # Same override for superheader — runtime accesses .attrs and
    # .template on it via with_defaults config.
    superheader = table.refinables["superheader"]
    assert superheader.kind == "class_ref"
    assert superheader.target == "iommi.table.HeaderConfig"

    # Column.header points at HeaderColumnConfig instead, which has
    # ``url`` and ``template`` but not ``tag``/``include``.
    col = g.get("iommi.table.Column")
    col_header = col.refinables["header"]
    assert col_header.kind == "class_ref"
    assert col_header.target == "iommi.table.HeaderColumnConfig"

    # BFS must have pulled both Config classes into the graph; their
    # refinables drive the per-segment validation downstream.
    assert g.has("iommi.table.HeaderConfig")
    assert g.has("iommi.table.HeaderColumnConfig")

    header_config = g.get("iommi.table.HeaderConfig")
    assert "template" in header_config.refinables
    assert "tag" in header_config.refinables
    assert "include" in header_config.refinables
    assert "attrs" in header_config.refinables
    assert header_config.refinables["attrs"].kind == "html_attrs"
    # ``extra``/``extra_evaluated`` get the open_namespace treatment via
    # the existing classifier hook.
    assert header_config.refinables["extra"].kind == "open_namespace"
    assert header_config.refinables["extra_evaluated"].kind == "open_namespace"

    header_col_config = g.get("iommi.table.HeaderColumnConfig")
    assert "template" in header_col_config.refinables
    assert "url" in header_col_config.refinables
    assert "attrs" in header_col_config.refinables
    # ``tag`` is unique to the table-level HeaderConfig.
    assert "tag" not in header_col_config.refinables


def test_collect_init_members_handles_decorated_init():
    """Cell's ``__init__`` is wrapped by ``@dispatch``. _collect_init_members
    must unwrap so the source can still be AST-parsed.
    """
    from iommi.table import Cell

    from iommi_lsp.analyzers.iommi.reflect import _collect_init_members

    names = _collect_init_members(Cell)
    assert "url" in names
    assert "value" in names
    assert "tag" in names
    # Private attributes are filtered out.
    assert not any(n.startswith("_") for n in names)
