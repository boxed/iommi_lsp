"""Static set of attributes that Django attaches to every model via
metaclass magic — the names ``ty`` will flag as ``unresolved-attribute``
when looking only at the user's source.

Kept here as a single source of truth so tests can introspect it and
the Django analyzer can use it without re-defining anything.
"""

from __future__ import annotations


# Manager-like accessors. ``objects`` is the obvious one; the others are
# present on every model regardless of explicit manager declarations.
MANAGER_ATTRS: frozenset[str] = frozenset({
    "objects",
    "_default_manager",
    "_base_manager",
})

# Meta / introspection.
META_ATTRS: frozenset[str] = frozenset({
    "_meta",
    "Meta",
})

# Primary-key aliases (always available; ``id`` only when no explicit PK).
PK_ATTRS: frozenset[str] = frozenset({"pk", "id"})

# Exception classes injected by ``ModelBase``.
EXCEPTION_ATTRS: frozenset[str] = frozenset({
    "DoesNotExist",
    "MultipleObjectsReturned",
})

# Field types whose declarations create reverse accessors on the target.
RELATION_FIELD_NAMES: frozenset[str] = frozenset({
    "ForeignKey",
    "OneToOneField",
    "ManyToManyField",
})

# Field types whose declarations create a ``<name>_id`` accessor on the
# declaring model. (``ManyToManyField`` does *not* — it goes through a
# through-table.)
FK_LIKE_FIELD_NAMES: frozenset[str] = frozenset({
    "ForeignKey",
    "OneToOneField",
})


# Aggregate of attributes that always exist on a Django model regardless
# of its declarations. Reverse relations and FK-id accessors are
# index-driven and not in this set.
ALWAYS_PRESENT: frozenset[str] = (
    MANAGER_ATTRS | META_ATTRS | PK_ATTRS | EXCEPTION_ATTRS
)
