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
    "choices_enum": frozenset(),           # drops ty's invalid-assignment on IntegerChoices/TextChoices tuple members
    "f_operator": frozenset(),             # drops ty's unsupported-operator on F()/Combinable arithmetic
}


@dataclass(frozen=True)
class Config:
    enabled: bool = True
    disabled_rules: frozenset[str] = field(default_factory=frozenset)
    extra_magic_attrs: dict[str, frozenset[str]] = field(default_factory=dict)

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

    section = (data.get("tool") or {}).get("iommi_lsp")
    if not isinstance(section, dict):
        return DEFAULT

    return _from_dict(section, source=str(pyproject))


def _from_dict(section: dict, *, source: str) -> Config:
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
    )
