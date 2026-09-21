"""Pure-function walker for Django ORM ``__`` lookup chains.

Given a starting model and the segments of a flattened ORM kwarg name
(split on ``__``), walks the field/relation graph and returns ``OK`` or
a ``Problem`` describing the first segment that didn't validate. Bias is
toward false negatives — unknown models, unknown relation targets,
custom field types, or anything past a known lookup operator → ``OK``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .index import DjangoIndex, ModelInfo
from .magic import ORM_LOOKUP_NAMES, RELATION_FIELD_NAMES


WalkOutcome = Literal["unknown_field", "unknown_lookup", "trailing_after_leaf"]


@dataclass(frozen=True)
class Problem:
    outcome: WalkOutcome
    bad_segment: str
    segment_index: int    # 0-based position in the chain
    on_model: str         # qualname of the model we were validating against
    available: tuple[str, ...] = ()  # known segments on `on_model` (for hints)


@dataclass(frozen=True)
class Ok:
    pass


OK = Ok()
WalkResult = Ok | Problem


def split_chain(kwarg_name: str) -> list[str]:
    return kwarg_name.split("__")


def walk(index: DjangoIndex, model_qualname: str, chain: list[str]) -> WalkResult:
    """Walk *chain* starting at *model_qualname*. Returns OK or first Problem."""
    if not chain:
        return OK
    model = index.models.get(model_qualname)
    if model is None:
        return OK   # unknown model → can't validate
    return _walk_from(index, model, chain, 0)


def _walk_from(
    index: DjangoIndex, model: ModelInfo, chain: list[str], i: int
) -> WalkResult:
    if i >= len(chain):
        return OK

    seg = chain[i]

    # `pk` — alias for primary key. Treat as leaf scalar. `id` is the
    # column Django adds itself when the model declares no explicit PK,
    # so `page__id=…` is as valid as `page__pk=…`.
    if seg == "pk" or (seg == "id" and model.implicit_id):
        return _step_after_leaf(model, chain, i + 1)

    # `<fk>_id` — explicit underlying-column accessor on FK/OneToOne.
    if seg in model.fk_id_accessors:
        return _step_after_leaf(model, chain, i + 1)

    fi = model.fields.get(seg)
    if fi is None:
        # Reverse relation?
        source = index.reverse_query_source(model.qualname, seg)
        if source is not None:
            # Reverse-relation + lookup (`tags__in=[…]`) acts as a leaf.
            if i + 1 < len(chain) and chain[i + 1] in ORM_LOOKUP_NAMES:
                return _step_after_leaf(model, chain, i + 1)
            target = index.models.get(source)
            if target is None:
                return OK
            return _walk_from(index, target, chain, i + 1)
        return Problem(
            outcome="unknown_field",
            bad_segment=seg,
            segment_index=i,
            on_model=model.qualname,
            available=tuple(sorted(_available_segments(index, model))),
        )

    # Concrete or relation field.
    if fi.field_type in RELATION_FIELD_NAMES:
        if i + 1 >= len(chain):
            # Terminal — `.filter(author=user)` is fine.
            return OK
        # `.filter(tags__in=[…])`, `.filter(author__isnull=True)` —
        # a relation followed by a lookup name acts as a leaf, with
        # Django filtering by PK/instance.
        if chain[i + 1] in ORM_LOOKUP_NAMES:
            return _step_after_leaf(model, chain, i + 1)
        target_qualname = fi.target
        if target_qualname is None or target_qualname not in index.models:
            return OK   # unknown target → bias toward OK
        return _walk_from(index, index.models[target_qualname], chain, i + 1)

    # Concrete (non-relation) field — leaf.
    return _step_after_leaf(model, chain, i + 1)


def _step_after_leaf(model: ModelInfo, chain: list[str], j: int) -> WalkResult:
    """We just consumed a leaf field; *j* is the next segment index."""
    if j >= len(chain):
        return OK
    seg = chain[j]
    if seg in ORM_LOOKUP_NAMES:
        # Transforms can chain (`pubdate__year__gte`) and custom fields
        # may register custom lookups — accept everything past the first
        # known lookup name.
        return OK
    return Problem(
        outcome="unknown_lookup",
        bad_segment=seg,
        segment_index=j,
        on_model=model.qualname,
    )


def _available_segments(index: DjangoIndex, model: ModelInfo) -> set[str]:
    names: set[str] = set(model.fields.keys())
    names.update(model.fk_id_accessors)
    names.add("pk")
    if model.implicit_id:
        names.add("id")
    names.update(index.reverse_query_names(model.qualname))
    return names
