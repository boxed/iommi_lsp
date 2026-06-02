"""Tests for the ``field_union`` suppression in :class:`DjangoAnalyzer`.

``Model._meta.get_fields()`` is typed (by django-stubs) as
``list[Field[Any, Any] | ForeignObjectRel]``. Iterating it and reaching for
a concrete-field attribute — ``attname``, ``column``, … — makes ty emit
``unresolved-attribute`` on the ``ForeignObjectRel`` arm of the union, even
though field-walking code never trips over it in practice. The filter drops
that noise while leaving genuinely-unresolved attributes alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iommi_lsp.analyzers.django import DjangoAnalyzer, build_index


def _attr_diag(message: str, code: str = "unresolved-attribute"):
    """Build an ``unresolved-attribute`` diagnostic carrying *message*.

    The filter is message-only, so the range is immaterial.
    """
    return {
        "code": code,
        "message": message,
        "range": {
            "start": {"line": 6, "character": 8},
            "end": {"line": 6, "character": 21},
        },
        "severity": 1,
        "source": "ty",
    }


@pytest.fixture
def analyzer(tmp_path: Path) -> DjangoAnalyzer:
    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)
    return a


def test_attname_on_foreignobjectrel_arm_is_dropped(analyzer: DjangoAnalyzer):
    """The exact shape ty emits for ``field.attname`` over get_fields()."""
    diag = _attr_diag(
        "Attribute `attname` is not defined on `ForeignObjectRel` "
        "in union `Field | ForeignObjectRel`"
    )
    assert analyzer.is_false_positive("file:///u.py", diag) is True


def test_generic_field_arm_with_params_is_dropped(analyzer: DjangoAnalyzer):
    """django-stubs spells the field arm with type params: ``Field[Any, Any]``."""
    diag = _attr_diag(
        "Attribute `column` is not defined on `ForeignObjectRel` "
        "in union `Field[Any, Any] | ForeignObjectRel`"
    )
    assert analyzer.is_false_positive("file:///u.py", diag) is True


@pytest.mark.parametrize(
    "rel",
    ["ManyToOneRel", "OneToOneRel", "ManyToManyRel", "GenericRel"],
)
def test_concrete_rel_subclasses_are_dropped(analyzer: DjangoAnalyzer, rel: str):
    diag = _attr_diag(
        f"Attribute `db_column` is not defined on `{rel}` "
        f"in union `Field | {rel}`"
    )
    assert analyzer.is_false_positive("file:///u.py", diag) is True


def test_missing_arm_not_a_relation_is_kept(analyzer: DjangoAnalyzer):
    """A union that happens to be ``Field | None`` is a different bug — keep it."""
    diag = _attr_diag(
        "Attribute `attname` is not defined on `None` in union `Field | None`"
    )
    assert analyzer.is_false_positive("file:///u.py", diag) is False


def test_union_without_field_arm_is_kept(analyzer: DjangoAnalyzer):
    """No ``Field`` present — not the get_fields() shape, so keep it."""
    diag = _attr_diag(
        "Attribute `attname` is not defined on `ForeignObjectRel` "
        "in union `str | ForeignObjectRel`"
    )
    assert analyzer.is_false_positive("file:///u.py", diag) is False


def test_non_union_unresolved_attribute_is_untouched(analyzer: DjangoAnalyzer):
    """A plain unresolved-attribute with no union falls through to the
    normal model-aware path (which keeps it for an unknown receiver)."""
    diag = _attr_diag("Object of type `Whatever` has no attribute `attname`")
    assert analyzer.is_false_positive("file:///u.py", diag) is False


def test_wrong_code_is_kept(analyzer: DjangoAnalyzer):
    """Same message text under a different code is not our diagnostic."""
    diag = _attr_diag(
        "Attribute `attname` is not defined on `ForeignObjectRel` "
        "in union `Field | ForeignObjectRel`",
        code="some-other-code",
    )
    assert analyzer.is_false_positive("file:///u.py", diag) is False


def test_disabled_rule_keeps_diagnostic(tmp_path: Path):
    """With ``field_union`` in disabled_rules the diagnostic survives."""
    from iommi_lsp.config import Config

    a = DjangoAnalyzer(
        workspace_root=tmp_path,
        config=Config(disabled_rules=frozenset({"field_union"})),
    )
    a.django_index = build_index(tmp_path)
    diag = _attr_diag(
        "Attribute `attname` is not defined on `ForeignObjectRel` "
        "in union `Field | ForeignObjectRel`"
    )
    assert a.is_false_positive("file:///u.py", diag) is False
