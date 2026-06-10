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

# Reverse-relation descriptor types Django mixes into ``_meta.get_fields()``
# alongside concrete ``Field`` instances. django-stubs types ``get_fields()``
# as ``list[Field[Any, Any] | ForeignObjectRel]``, so iterating and reaching
# for a concrete-field attribute (``attname``, ``column``, …) trips ty's
# ``unresolved-attribute`` on the ``ForeignObjectRel`` arm of the union.
FIELD_UNION_REL_NAMES: frozenset[str] = frozenset({
    "ForeignObjectRel",
    "ManyToOneRel",
    "OneToOneRel",
    "ManyToManyRel",
    "GenericRel",
})

# Field types whose declarations create a ``<name>_id`` accessor on the
# declaring model. (``ManyToManyField`` does *not* — it goes through a
# through-table.)
FK_LIKE_FIELD_NAMES: frozenset[str] = frozenset({
    "ForeignKey",
    "OneToOneField",
})


# Date / time field types — declarations like these gain auto-generated
# ``get_next_by_<name>()`` / ``get_previous_by_<name>()`` methods on the
# concrete model. (Django adds them in :class:`ModelBase` whenever a
# concrete model declares a non-null date/datetime field; we accept
# nullable ones too — Django still injects the methods, they just raise
# at runtime when there's no anchor instance.)
DATE_FIELD_NAMES: frozenset[str] = frozenset({
    "DateField",
    "DateTimeField",
})


# Django HTTP response classes. They all inherit ``__setitem__`` from
# ``HttpResponseBase``, which stringifies whatever value it's handed
# (``_convert_to_charset`` does ``str(value)`` for non-str/bytes). So
# ``response[header] = value`` is runtime-valid for any value type, but
# django-stubs types the setter as ``(str, str | bytes | int)`` — ty then
# flags ``response['X'] = guess_type(p)[0]`` (a ``str | None``) and similar
# as ``invalid-assignment``. We suppress those when the receiver's type is
# one of these.
DJANGO_RESPONSE_TYPE_NAMES: frozenset[str] = frozenset({
    "HttpResponseBase",
    "HttpResponse",
    "StreamingHttpResponse",
    "FileResponse",
    "JsonResponse",
    "HttpResponseRedirect",
    "HttpResponsePermanentRedirect",
    "HttpResponseRedirectBase",
    "HttpResponseNotModified",
    "HttpResponseBadRequest",
    "HttpResponseNotFound",
    "HttpResponseForbidden",
    "HttpResponseNotAllowed",
    "HttpResponseGone",
    "HttpResponseServerError",
    "TemplateResponse",
    "SimpleTemplateResponse",
})


# Aggregate of attributes that always exist on a Django model regardless
# of its declarations. Reverse relations and FK-id accessors are
# index-driven and not in this set.
ALWAYS_PRESENT: frozenset[str] = (
    MANAGER_ATTRS | META_ATTRS | PK_ATTRS | EXCEPTION_ATTRS
)


# Built-in ORM lookups + transforms recognised after a leaf field in a
# ``filter()/exclude()/get()`` chain. Once we hit one of these we stop
# validating (transforms can chain — e.g. ``pubdate__year__gte``) and let
# everything past it through.
ORM_LOOKUP_NAMES: frozenset[str] = frozenset({
    # Comparison.
    "exact", "iexact", "contains", "icontains", "in", "gt", "gte",
    "lt", "lte", "startswith", "istartswith", "endswith", "iendswith",
    "range", "isnull", "regex", "iregex", "search",
    # Date/time transforms.
    "year", "iso_year", "month", "day", "week", "week_day",
    "iso_week_day", "quarter", "hour", "minute", "second",
    "date", "time",
    # Postgres array/JSON.
    "overlap", "contained_by", "contains_any", "has_key", "has_keys",
    "has_any_keys",
})
