"""Per-project configuration via ``[tool.iommi_lsp]`` in ``pyproject.toml``.

Schema:

.. code-block:: toml

    [tool.iommi_lsp]
    enabled = true                              # master switch
    disabled_rules = ["pk", "reverse"]           # skip these rule groups
    extra_magic_attrs = { manager = ["mongo"] }  # add to a group

Recognised rule groups: ``manager``, ``meta``, ``pk``, ``exception``,
``fk_id``, ``reverse``. Anything unknown is ignored with a warning so
typos in the config don't silently neutralise the filter.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — we require 3.11+ in pyproject anyway
    import tomli as tomllib

from . import log
from .analyzers.django.magic import (
    EXCEPTION_ATTRS,
    MANAGER_ATTRS,
    META_ATTRS,
    PK_ATTRS,
)


_log = log.get("config")


# Rule group → the static set of attrs it covers. Reverse / fk_id are
# index-driven and don't have a static set, but they still appear here
# so the disabled_rules switch knows about them.
RULE_GROUPS: dict[str, frozenset[str]] = {
    "manager": MANAGER_ATTRS,
    "meta": META_ATTRS,
    "pk": PK_ATTRS,
    "exception": EXCEPTION_ATTRS,
    "fk_id": frozenset(),                  # dynamic — see ModelInfo.fk_id_accessors
    "reverse": frozenset(),                # dynamic — see DjangoIndex.reverse_relations
    "generated": frozenset(),              # dynamic — see ModelInfo.generated_method_names
    "orm_lookup": frozenset(),             # dynamic — see DjangoAnalyzer.additional_diagnostics
    "unused_request_param": frozenset(),   # drops ty's "`request` is unused" hint when it's the first param
    "unused_test_param": frozenset(),      # drops ty's "`x` is unused" hint on a parameter of a test function, a pytest fixture, or any callable in a test module
    "unused_param_in_stub": frozenset(),   # drops ty's "`x` is unused" hint on a parameter of a stub body (``...``, ``pass``, ``raise NotImplementedError``)
    "unused_varargs": frozenset(),         # drops ty's "`x` is unused" hint on a `*args` / `**kwargs` parameter (they exist to absorb, not to be read)
    "unused_unpacked_name": frozenset(),   # drops ty's "`x` is unused" hint on a name bound by sequence unpacking (you can't leave one out)
    "function_attribute": frozenset(),     # drops ty's unresolved-attribute when the receiver is a function object (decorators that return a wrapper add attributes ty can't see)
    "choices_enum": frozenset(),           # drops ty's invalid-assignment on IntegerChoices/TextChoices tuple members
    "f_operator": frozenset(),             # drops ty's unsupported-operator on F()/Combinable arithmetic
    "annotate": frozenset(),               # drops ty's unresolved-attribute on annotate()/alias() aliases
    "relation_field_assignment": frozenset(),  # drops ty's invalid-assignment on ForeignKey/OneToOneField/ManyToManyField field declarations
    "scalar_field_assignment": frozenset(),  # drops ty's invalid-assignment on scalar field declarations (CharField→str, IntegerField→int, …) when the target type matches the field's Python type
    "field_union": frozenset(),            # drops ty's unresolved-attribute on the ForeignObjectRel arm of _meta.get_fields()'s Field|ForeignObjectRel union
    "decorator_typevar": frozenset(),      # drops ty's invalid-argument-type ("Expected TypeVar") on require_POST/require_GET-style passthrough decorators
    "response_header_assignment": frozenset(),  # drops ty's invalid-assignment on `response[header] = value` (HttpResponseBase.__setitem__ stringifies any value)
    "relation_field_attribute": frozenset(),  # drops ty's unresolved-attribute when the receiver is a ForeignKey/OneToOneField/ManyToManyField (the descriptor yields the related instance)
    "field_attribute_assignment": frozenset(),  # drops ty's invalid-assignment onto an instance attribute whose declared type is a model field
    "field_operator": frozenset(),         # drops ty's unsupported-operator when an operand's type is a model field (the descriptor yields the Python value)
    "field_argument_type": frozenset(),    # drops ty's invalid-argument-type when the argument's type is a model field
    "splat_argument": frozenset(),         # drops ty's argument-shape diagnostics (invalid-argument-type, no-matching-overload, …) for a call that passes `**some_dict`
    "field_narrowing_unreachable": frozenset(),  # drops ty's "Code is unreachable" when an earlier guard narrows on a model-field attribute (a field class object is never None / always truthy to ty)
}


@dataclass(frozen=True)
class Config:
    enabled: bool = True
    disabled_rules: frozenset[str] = field(default_factory=frozenset)
    extra_magic_attrs: dict[str, frozenset[str]] = field(default_factory=dict)
    # ``[tool.ty.analysis.names-to-types]`` — not our section, but ty
    # reads it to type unannotated names by convention (``project`` is a
    # ``Project``, ``user`` a ``User``, …). Reading it too lets our AST
    # resolvers see the same types ty does in test files and other
    # un-annotated code. Name → the qualnames it maps to (``A|B`` unions
    # keep both arms).
    names_to_types: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def is_rule_enabled(self, rule: str) -> bool:
        return self.enabled and rule not in self.disabled_rules

    def merged_static_attrs(self, group: str) -> frozenset[str]:
        """Return the static attrs for *group* with config additions merged in."""
        base = RULE_GROUPS.get(group, frozenset())
        extra = self.extra_magic_attrs.get(group, frozenset())
        return base | extra


DEFAULT = Config()


def load(workspace_root: Path) -> Config:
    """Read ``pyproject.toml`` at *workspace_root* and parse our section.

    Missing file or missing section -> return :data:`DEFAULT`. A malformed
    file is treated the same way (with a warning) — we never want a
    config typo to break the proxy.
    """
    pyproject = workspace_root / "pyproject.toml"
    if not pyproject.exists():
        return DEFAULT
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        _log.warning("could not parse %s: %s; using defaults", pyproject, e)
        return DEFAULT

    names_to_types = _read_names_to_types(data, source=str(pyproject))

    section = (data.get("tool") or {}).get("iommi_lsp")
    if not isinstance(section, dict):
        if not names_to_types:
            return DEFAULT
        return Config(names_to_types=names_to_types)

    return _from_dict(
        section, source=str(pyproject), names_to_types=names_to_types,
    )


def _read_names_to_types(data: dict, *, source: str) -> dict[str, tuple[str, ...]]:
    """Parse ``[tool.ty.analysis.names-to-types]`` (a ty-semantic feature)."""
    analysis = ((data.get("tool") or {}).get("ty") or {}).get("analysis")
    if not isinstance(analysis, dict):
        return {}
    raw = analysis.get("names-to-types")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for name, value in raw.items():
        if not isinstance(value, str):
            _log.warning("%s: names-to-types.%s must be a string", source, name)
            continue
        arms = tuple(a.strip() for a in value.split("|") if a.strip())
        if arms:
            out[name] = arms
    return out


def _from_dict(
    section: dict,
    *,
    source: str,
    names_to_types: dict[str, tuple[str, ...]] | None = None,
) -> Config:
    enabled = bool(section.get("enabled", True))

    raw_disabled = section.get("disabled_rules") or []
    disabled: set[str] = set()
    if isinstance(raw_disabled, list):
        for item in raw_disabled:
            if not isinstance(item, str):
                _log.warning("%s: disabled_rules entries must be strings, got %r", source, item)
                continue
            if item not in RULE_GROUPS:
                _log.warning(
                    "%s: unknown rule group %r in disabled_rules; valid: %s",
                    source, item, sorted(RULE_GROUPS),
                )
                continue
            disabled.add(item)
    else:
        _log.warning("%s: disabled_rules must be a list", source)

    raw_extra = section.get("extra_magic_attrs") or {}
    extra: dict[str, frozenset[str]] = {}
    if isinstance(raw_extra, dict):
        for group, attrs in raw_extra.items():
            if group not in RULE_GROUPS:
                _log.warning(
                    "%s: unknown rule group %r in extra_magic_attrs",
                    source, group,
                )
                continue
            if not isinstance(attrs, list) or not all(isinstance(a, str) for a in attrs):
                _log.warning(
                    "%s: extra_magic_attrs.%s must be a list of strings",
                    source, group,
                )
                continue
            extra[group] = frozenset(attrs)
    else:
        _log.warning("%s: extra_magic_attrs must be a table", source)

    return Config(
        enabled=enabled,
        disabled_rules=frozenset(disabled),
        extra_magic_attrs=extra,
        names_to_types=names_to_types or {},
    )
