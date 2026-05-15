"""Pure-function tests for the chain walker."""

from __future__ import annotations

import pytest

from iommi_lsp.analyzers.iommi.graph import IommiClass, IommiGraph, Refinable
from iommi_lsp.analyzers.iommi.walker import OK, Problem, walk


def _r(name: str, kind: str, **kw) -> Refinable:
    return Refinable(name=name, kind=kind, **kw)


@pytest.fixture
def graph() -> IommiGraph:
    """Hand-built minimal graph with one of each refinable kind."""
    column = IommiClass(
        qualname="x.Column",
        bases=["x.Part"],
        refinables={
            "extra": _r("extra", "open_namespace"),
            "cell": _r("cell", "namespace", known_keys=["attrs", "contents"]),
            "after": _r("after", "evaluated_scalar"),
        },
    )
    part = IommiClass(
        qualname="x.Part",
        bases=[],
        refinables={
            "attrs": _r(
                "attrs", "html_attrs",
                sub_specials={"class": {"value_type": "bool"}, "style": {"value_type": "str"}},
            ),
        },
    )
    table = IommiClass(
        qualname="x.Table",
        bases=["x.Part"],
        refinables={
            "columns": _r("columns", "members", member_class="x.Column"),
            "parts": _r("parts", "members"),
            "page_size": _r("page_size", "evaluated_scalar"),
            "bulk": _r("bulk", "class_ref", target="x.Form"),
            "extra": _r("extra", "open_namespace"),
            "attrs": _r(
                "attrs", "html_attrs",
                sub_specials={"class": {"value_type": "bool"}, "style": {"value_type": "str"}},
            ),
        },
    )
    form = IommiClass(
        qualname="x.Form",
        bases=[],
        refinables={
            "fields": _r("fields", "members"),
            "title": _r("title", "evaluated_scalar"),
        },
    )
    return IommiGraph(classes={c.qualname: c for c in [table, column, part, form]})


def test_known_top_level_refinable_passes(graph):
    assert walk(graph, "x.Table", ["page_size"]) is OK


def test_unknown_top_level_refinable_fails(graph):
    res = walk(graph, "x.Table", ["bogus_thing"])
    assert isinstance(res, Problem)
    assert res.outcome == "unknown_refinable"
    assert res.bad_segment == "bogus_thing"
    assert res.segment_index == 0
    assert res.on_class == "x.Table"


def test_chain_through_members_validates_member_class(graph):
    # columns -> any name -> step into Column refinables
    assert walk(graph, "x.Table", ["columns", "name", "after"]) is OK


def test_unknown_refinable_on_member_class(graph):
    res = walk(graph, "x.Table", ["columns", "name", "bogus"])
    assert isinstance(res, Problem)
    assert res.bad_segment == "bogus"
    assert res.on_class == "x.Column"


def test_chain_through_open_namespace_accepts_anything(graph):
    assert walk(graph, "x.Table", ["extra", "anything", "deep", "key"]) is OK


def test_namespace_known_keys_validate(graph):
    assert walk(graph, "x.Table", ["columns", "name", "cell", "attrs", "anything"]) is OK


def test_namespace_unknown_key_fails(graph):
    res = walk(graph, "x.Table", ["columns", "name", "cell", "bogus"])
    assert isinstance(res, Problem)
    assert res.bad_segment == "bogus"
    assert res.outcome == "unknown_refinable"


def test_class_ref_steps_into_target(graph):
    # bulk -> Form. fields is on Form. Should validate.
    assert walk(graph, "x.Table", ["bulk", "fields", "name"]) is OK


def test_class_ref_unknown_refinable_on_target(graph):
    res = walk(graph, "x.Table", ["bulk", "bogus_form_thing"])
    assert isinstance(res, Problem)
    assert res.on_class == "x.Form"
    assert res.bad_segment == "bogus_form_thing"


def test_html_attrs_direct_attribute_ok(graph):
    assert walk(graph, "x.Table", ["attrs", "data_thing"]) is OK


def test_html_attrs_class_subspecial(graph):
    assert walk(graph, "x.Table", ["attrs", "class", "bold"]) is OK


def test_html_attrs_style_subspecial(graph):
    assert walk(graph, "x.Table", ["attrs", "style", "color"]) is OK


def test_html_attrs_chain_past_class_value_fails(graph):
    res = walk(graph, "x.Table", ["attrs", "class", "bold", "extra"])
    assert isinstance(res, Problem)
    assert res.outcome == "trailing_segments_after_leaf"
    assert res.bad_segment == "extra"


def test_html_attrs_chain_past_direct_attribute_fails(graph):
    res = walk(graph, "x.Table", ["attrs", "data_thing", "deeper"])
    assert isinstance(res, Problem)
    assert res.outcome == "trailing_segments_after_leaf"


def test_chain_past_scalar_fails(graph):
    res = walk(graph, "x.Table", ["page_size", "bogus"])
    assert isinstance(res, Problem)
    assert res.outcome == "trailing_segments_after_leaf"
    assert res.bad_segment == "bogus"


def test_inherited_refinable_via_base_class(graph):
    # `attrs` is on Part; Column inherits Part. Validating Column.attrs OK.
    assert walk(graph, "x.Column", ["attrs", "data_x"]) is OK


def test_unknown_root_class_silently_passes(graph):
    # Not in graph -> walker bias toward false negatives.
    assert walk(graph, "x.Unknown", ["bogus", "more"]) is OK


def test_members_with_no_member_class_accepts_anything(graph):
    # `parts` is members with no member_class -> open after the user key.
    assert walk(graph, "x.Table", ["parts", "anything", "deeper"]) is OK


# ---------------------------------------------------------------------------
# `attrs` heuristic: any segment named ``attrs`` reached from an iommi
# class behaves like html_attrs, even when the static reflector missed it.
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_no_part_attrs() -> IommiGraph:
    """Like `graph` but Column doesn't inherit `attrs` (Part is missing)."""
    column = IommiClass(
        qualname="x.Column",
        bases=[],   # no base classes in graph at all
        refinables={
            "extra": _r("extra", "open_namespace"),
            "cell": _r("cell", "namespace", known_keys=["attrs", "contents", "link"]),
        },
    )
    table = IommiClass(
        qualname="x.Table",
        bases=[],
        refinables={
            "columns": _r("columns", "members", member_class="x.Column"),
        },
    )
    return IommiGraph(classes={c.qualname: c for c in [table, column]})


def test_attrs_on_class_without_declared_refinable_is_treated_as_html_attrs(graph_no_part_attrs):
    # Column has no `attrs` refinable in this graph at all. The walker
    # should still accept `columns__name__attrs__class__bold` because
    # `attrs` is the universal iommi escape hatch.
    assert walk(graph_no_part_attrs, "x.Table", ["columns", "name", "attrs", "class", "bold"]) is OK


def test_attrs_inside_namespace_recurses_into_html_attrs(graph_no_part_attrs):
    # cell is a namespace with known_keys [attrs, contents, link]. Stepping
    # through `cell__attrs__class__bold` should validate via html_attrs.
    assert walk(graph_no_part_attrs, "x.Table", ["columns", "name", "cell", "attrs", "class", "bold"]) is OK


def test_attrs_inside_namespace_chain_past_class_value_fails(graph_no_part_attrs):
    res = walk(graph_no_part_attrs, "x.Table", [
        "columns", "name", "cell", "attrs", "class", "bold", "deeper"
    ])
    assert isinstance(res, Problem)
    assert res.outcome == "trailing_segments_after_leaf"


def test_attrs_inside_namespace_direct_attribute_ok(graph_no_part_attrs):
    # cell.attrs.data_thing="hi" -> direct HTML attribute, not class/style.
    assert walk(graph_no_part_attrs, "x.Table", ["columns", "name", "cell", "attrs", "data_thing"]) is OK


def test_non_attrs_namespace_keys_still_pass_through_freely(graph_no_part_attrs):
    # cell.contents is a known key but not "attrs" -> existing permissive
    # behavior (no further validation).
    assert walk(graph_no_part_attrs, "x.Table", ["columns", "name", "cell", "contents", "anything"]) is OK


# ---------------------------------------------------------------------------
# traditional_class — refinable that drills into a non-Refinable class via
# its ``__init__`` self-assignments. Used for ``Column.cell`` etc.
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_with_traditional() -> IommiGraph:
    column = IommiClass(
        qualname="x.Column",
        bases=[],
        refinables={
            "cell": _r("cell", "traditional_class", target="x.Cell"),
        },
    )
    cell = IommiClass(
        qualname="x.Cell",
        bases=[],
        refinables={},
        init_members=["url", "url_title", "value", "tag"],
    )
    table = IommiClass(
        qualname="x.Table",
        bases=[],
        refinables={
            "columns": _r("columns", "members", member_class="x.Column"),
        },
    )
    return IommiGraph(classes={c.qualname: c for c in [table, column, cell]})


def test_traditional_class_known_init_member_ok(graph_with_traditional):
    assert walk(graph_with_traditional, "x.Table",
                ["columns", "name", "cell", "url"]) is OK


def test_traditional_class_unknown_init_member_fails(graph_with_traditional):
    res = walk(graph_with_traditional, "x.Table",
               ["columns", "name", "cell", "bogus"])
    assert isinstance(res, Problem)
    assert res.outcome == "unknown_refinable"
    assert res.bad_segment == "bogus"
    assert res.on_class == "x.Cell"
    assert "url" in res.available


def test_traditional_class_trailing_after_init_member_fails(graph_with_traditional):
    res = walk(graph_with_traditional, "x.Table",
               ["columns", "name", "cell", "url", "deeper"])
    assert isinstance(res, Problem)
    assert res.outcome == "trailing_segments_after_leaf"
    assert res.bad_segment == "deeper"
    assert res.on_class == "x.Cell"


def test_traditional_class_without_init_members_passes(graph_with_traditional):
    """When the graph doesn't know the target's init members, bias toward OK."""
    g = IommiGraph(classes={
        "x.Column": IommiClass(
            qualname="x.Column", bases=[],
            refinables={"cell": _r("cell", "traditional_class", target="x.NoSuch")},
        ),
    })
    assert walk(g, "x.Column", ["cell", "anything"]) is OK


@pytest.fixture
def graph_with_traditional_attrs() -> IommiGraph:
    """Cell.init_members includes ``attrs`` — the html_attrs namespace."""
    column = IommiClass(
        qualname="x.Column",
        bases=[],
        refinables={
            "cell": _r("cell", "traditional_class", target="x.Cell"),
        },
    )
    cell = IommiClass(
        qualname="x.Cell",
        bases=[],
        refinables={},
        init_members=["attrs", "url", "url_title", "value", "tag"],
    )
    return IommiGraph(classes={c.qualname: c for c in [column, cell]})


def test_traditional_attrs_recurses_into_html_attrs(graph_with_traditional_attrs):
    # `Column(cell__attrs__class__pre_wrap=True)` — `attrs` is an init_member
    # of Cell but should be treated as the html_attrs namespace, not a leaf.
    assert walk(graph_with_traditional_attrs, "x.Column",
                ["cell", "attrs", "class", "pre_wrap"]) is OK


def test_traditional_attrs_style_key_ok(graph_with_traditional_attrs):
    assert walk(graph_with_traditional_attrs, "x.Column",
                ["cell", "attrs", "style", "color"]) is OK


def test_traditional_attrs_direct_attribute_ok(graph_with_traditional_attrs):
    # cell.attrs.data_thing="hi" — direct HTML attribute leaf.
    assert walk(graph_with_traditional_attrs, "x.Column",
                ["cell", "attrs", "data_thing"]) is OK


def test_traditional_attrs_chain_past_class_value_fails(graph_with_traditional_attrs):
    res = walk(graph_with_traditional_attrs, "x.Column",
               ["cell", "attrs", "class", "pre_wrap", "deeper"])
    assert isinstance(res, Problem)
    assert res.outcome == "trailing_segments_after_leaf"


def test_traditional_attrs_chain_past_direct_attribute_fails(graph_with_traditional_attrs):
    res = walk(graph_with_traditional_attrs, "x.Column",
               ["cell", "attrs", "data_thing", "deeper"])
    assert isinstance(res, Problem)
    assert res.outcome == "trailing_segments_after_leaf"


def test_traditional_non_attrs_init_member_still_a_leaf(graph_with_traditional_attrs):
    # `tag` is a normal init_member — chaining past it is still invalid.
    res = walk(graph_with_traditional_attrs, "x.Column",
               ["cell", "tag", "extra"])
    assert isinstance(res, Problem)
    assert res.outcome == "trailing_segments_after_leaf"


# ---------------------------------------------------------------------------
# `attrs` after a misclassified scalar/evaluated_scalar refinable. Iommi
# declares things like ``header: Namespace = EvaluatedRefinable()``; static
# reflection collapses this to ``evaluated_scalar`` and the walker would
# otherwise reject the canonical ``header__attrs__class__numeric=True``
# usage. ``attrs`` is iommi's universal escape hatch, so we recurse into
# html_attrs regardless of how the parent refinable was classified.
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_misclassified_scalar_with_attrs() -> IommiGraph:
    """Column.header looks like an evaluated_scalar leaf in the graph, but
    iommi actually unpacks it into a ``HeaderColumnConfig`` with its own
    ``attrs`` namespace."""
    column = IommiClass(
        qualname="x.Column",
        bases=[],
        refinables={
            "header": _r("header", "evaluated_scalar"),
            "superheader": _r("superheader", "scalar"),
        },
    )
    return IommiGraph(classes={c.qualname: c for c in [column]})


def test_attrs_after_evaluated_scalar_treated_as_html_attrs(
    graph_misclassified_scalar_with_attrs,
):
    # Column(header__attrs__class__numeric=True) — the user's canonical
    # idiom. Must not warn.
    assert walk(graph_misclassified_scalar_with_attrs, "x.Column",
                ["header", "attrs", "class", "numeric"]) is OK


def test_attrs_after_scalar_treated_as_html_attrs(
    graph_misclassified_scalar_with_attrs,
):
    assert walk(graph_misclassified_scalar_with_attrs, "x.Column",
                ["superheader", "attrs", "class", "bold"]) is OK


def test_attrs_after_scalar_style_subspecial_ok(
    graph_misclassified_scalar_with_attrs,
):
    assert walk(graph_misclassified_scalar_with_attrs, "x.Column",
                ["header", "attrs", "style", "color"]) is OK


def test_attrs_after_scalar_chain_past_class_value_still_fails(
    graph_misclassified_scalar_with_attrs,
):
    # The html_attrs leaf rules still apply once we recurse.
    res = walk(graph_misclassified_scalar_with_attrs, "x.Column",
               ["header", "attrs", "class", "numeric", "deeper"])
    assert isinstance(res, Problem)
    assert res.outcome == "trailing_segments_after_leaf"


def test_non_attrs_chain_past_scalar_still_fails(
    graph_misclassified_scalar_with_attrs,
):
    # Only ``attrs`` gets the recursion; an unrelated segment past a scalar
    # leaf is still a real chain error.
    res = walk(graph_misclassified_scalar_with_attrs, "x.Column",
               ["header", "bogus"])
    assert isinstance(res, Problem)
    assert res.outcome == "trailing_segments_after_leaf"
    assert res.bad_segment == "bogus"


# ---------------------------------------------------------------------------
# ``Table.header`` / ``Column.header`` after the reflect-time override
# promotes them to ``class_ref`` against HeaderConfig / HeaderColumnConfig.
# These tests pin the walker's behavior so a regression in the override
# (or a structural rename in iommi) gets caught here, separately from the
# integration tests against the real graph.
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_with_header_class_ref() -> IommiGraph:
    """Mirror the post-override shape: Table.header / Column.header /
    Table.superheader all class_ref into header config classes that
    declare a realistic field set."""
    header_config = IommiClass(
        qualname="x.HeaderConfig",
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
        qualname="x.HeaderColumnConfig",
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
        qualname="x.Column",
        bases=[],
        refinables={
            "header": _r("header", "class_ref", target="x.HeaderColumnConfig"),
        },
    )
    table = IommiClass(
        qualname="x.Table",
        bases=[],
        refinables={
            "columns": _r("columns", "members", member_class="x.Column"),
            "header": _r("header", "class_ref", target="x.HeaderConfig"),
            "superheader": _r("superheader", "class_ref", target="x.HeaderConfig"),
        },
    )
    return IommiGraph(classes={c.qualname: c for c in [
        table, column, header_config, header_column_config,
    ]})


def test_header_class_ref_template_ok(graph_with_header_class_ref):
    """``Table(header__template=...)`` — the user-reported regression."""
    assert walk(graph_with_header_class_ref, "x.Table",
                ["header", "template"]) is OK


def test_header_class_ref_tag_ok(graph_with_header_class_ref):
    assert walk(graph_with_header_class_ref, "x.Table",
                ["header", "tag"]) is OK


def test_header_class_ref_include_ok(graph_with_header_class_ref):
    assert walk(graph_with_header_class_ref, "x.Table",
                ["header", "include"]) is OK


def test_header_class_ref_extra_open_namespace_ok(graph_with_header_class_ref):
    """``extra``/``extra_evaluated`` accept any sub-key."""
    assert walk(graph_with_header_class_ref, "x.Table",
                ["header", "extra", "any_user_key"]) is OK
    assert walk(graph_with_header_class_ref, "x.Table",
                ["header", "extra_evaluated", "any_user_key"]) is OK


def test_header_class_ref_attrs_class_ok(graph_with_header_class_ref):
    assert walk(graph_with_header_class_ref, "x.Table",
                ["header", "attrs", "class", "bold"]) is OK


def test_header_class_ref_attrs_direct_ok(graph_with_header_class_ref):
    assert walk(graph_with_header_class_ref, "x.Table",
                ["header", "attrs", "data_section"]) is OK


def test_header_class_ref_unknown_subkey_fails(graph_with_header_class_ref):
    res = walk(graph_with_header_class_ref, "x.Table",
               ["header", "bogus"])
    assert isinstance(res, Problem)
    assert res.outcome == "unknown_refinable"
    assert res.bad_segment == "bogus"
    assert res.on_class == "x.HeaderConfig"
    assert "template" in res.available


def test_header_class_ref_trailing_past_template_fails(graph_with_header_class_ref):
    """``template`` is a scalar leaf inside HeaderConfig — nothing past it."""
    res = walk(graph_with_header_class_ref, "x.Table",
               ["header", "template", "nope"])
    assert isinstance(res, Problem)
    assert res.outcome == "trailing_segments_after_leaf"
    assert res.bad_segment == "nope"


def test_superheader_class_ref_template_ok(graph_with_header_class_ref):
    """Table.superheader points at the same HeaderConfig — same surface."""
    assert walk(graph_with_header_class_ref, "x.Table",
                ["superheader", "template"]) is OK
    assert walk(graph_with_header_class_ref, "x.Table",
                ["superheader", "attrs", "class", "superheader"]) is OK


def test_column_header_class_ref_url_ok(graph_with_header_class_ref):
    """HeaderColumnConfig has ``url`` but no ``tag`` — verify the right
    config class is consulted for ``Column.header`` vs ``Table.header``."""
    assert walk(graph_with_header_class_ref, "x.Column",
                ["header", "url"]) is OK


def test_column_header_class_ref_template_ok(graph_with_header_class_ref):
    assert walk(graph_with_header_class_ref, "x.Column",
                ["header", "template"]) is OK


def test_column_header_class_ref_unknown_tag_fails(graph_with_header_class_ref):
    """``tag`` lives on HeaderConfig only — flagged on Column.header."""
    res = walk(graph_with_header_class_ref, "x.Column",
               ["header", "tag"])
    assert isinstance(res, Problem)
    assert res.outcome == "unknown_refinable"
    assert res.on_class == "x.HeaderColumnConfig"
    assert "url" in res.available


def test_column_header_class_ref_through_columns_chain(graph_with_header_class_ref):
    """``columns__name__header__template`` — the full chain through a
    members refinable into the per-column header config."""
    assert walk(graph_with_header_class_ref, "x.Table",
                ["columns", "name", "header", "template"]) is OK
    assert walk(graph_with_header_class_ref, "x.Table",
                ["columns", "name", "header", "url"]) is OK
    assert walk(graph_with_header_class_ref, "x.Table",
                ["columns", "name", "header", "attrs", "class", "x"]) is OK
