"""Import-time reflection of iommi's class structure.

Imports iommi (and minimally bootstraps Django so that import succeeds),
then walks each seed class's ``Refinable``-typed class attributes. The
walk is BFS-transitive: every ``class_ref`` target and ``members``
member_class is queued for its own pass, so the resulting graph is
closed under refinable-following.

This module is meant to be invoked from the *user's* venv via
``iommi_lsp graph build`` so the graph reflects the iommi version they
actually depend on, plus any subclasses they ship as installed packages.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import sys
import textwrap
from typing import Any

from .graph import IommiClass, IommiGraph, Refinable


# Manual overrides for refinables that look like a generic ``Namespace`` or
# scalar slot but in iommi's runtime actually configure a specific
# *traditional* (non-RefinableObject) class. The completion logic uses
# the target's ``init_members`` as the next chain step.
_TRADITIONAL_TARGETS: dict[str, dict[str, str]] = {
    "iommi.table.Column": {"cell": "iommi.table.Cell"},
    "iommi.table.Table": {"cell": "iommi.table.Cell"},
}


# Default seed: the names you'd reach via ``from iommi import ...``.
DEFAULT_SEEDS: tuple[str, ...] = (
    "iommi.Table",
    "iommi.Column",
    "iommi.Form",
    "iommi.Field",
    "iommi.Page",
    "iommi.Action",
    "iommi.Filter",
    "iommi.Fragment",
    "iommi.Asset",
    "iommi.MenuItem",
    "iommi.Menu",
)


def bootstrap_django() -> None:
    """Configure a minimal Django settings if the user hasn't already.

    iommi refuses to import unless ``django.setup()`` has run with iommi
    in ``INSTALLED_APPS``. Honour ``DJANGO_SETTINGS_MODULE`` if it's
    set; otherwise build the smallest viable settings ourselves.
    """
    import django
    from django.conf import settings

    if settings.configured:
        django.setup()
        return

    if os.environ.get("DJANGO_SETTINGS_MODULE"):
        django.setup()
        return

    settings.configure(
        USE_I18N=False,
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "iommi",
        ],
        DATABASES={
            "default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"},
        },
        SECRET_KEY="iommi_lsp-graph-build",
        ROOT_URLCONF=None,
        MIDDLEWARE=[],
    )
    django.setup()


def _qual(c: type) -> str:
    return f"{c.__module__}.{c.__qualname__}"


def _import_seed(spec: str) -> type | None:
    """Resolve ``"pkg.mod.Class"`` (or ``"pkg.Class"``) to a class object."""
    if "." not in spec:
        return None
    module_part, _, name = spec.rpartition(".")
    try:
        mod = importlib.import_module(module_part)
    except ImportError:
        return None
    return getattr(mod, name, None)


def _annotation_member_class(annotation: Any) -> str | None:
    """Extract the value type from a ``Dict[str, X]`` annotation."""
    from typing import get_args, get_origin

    if get_origin(annotation) is dict:
        args = get_args(annotation)
        if len(args) == 2 and isinstance(args[1], type):
            return _qual(args[1])
    return None


def _annotation_refinable_class(annotation: Any) -> str | None:
    """If the annotation refers to a single ``RefinableObject`` subclass
    (possibly wrapped in ``Optional``/``Union``), return its qualname."""
    from typing import Union, get_args, get_origin

    from iommi.refinable import RefinableObject

    candidates: list[type] = []
    if isinstance(annotation, type):
        candidates.append(annotation)
    elif get_origin(annotation) is Union:
        for arg in get_args(annotation):
            if isinstance(arg, type) and arg is not type(None):
                candidates.append(arg)

    refinable_classes = [c for c in candidates if issubclass(c, RefinableObject)]
    if len(refinable_classes) == 1:
        return _qual(refinable_classes[0])
    return None


def _classify(name: str, refinable_obj: Any, default_value: Any, annotation: Any) -> Refinable:
    from iommi.declarative.namespace import Namespace
    from iommi.refinable import (
        EvaluatedRefinable,
        RefinableMembers,
        SpecialEvaluatedRefinable,
    )

    rtype = type(refinable_obj).__name__

    if isinstance(refinable_obj, RefinableMembers):
        return Refinable(
            name=name,
            kind="members",
            refinable_type=rtype,
            member_class=_annotation_member_class(annotation),
        )

    if name == "attrs" and isinstance(refinable_obj, SpecialEvaluatedRefinable):
        return Refinable(
            name=name,
            kind="html_attrs",
            refinable_type=rtype,
            sub_specials={
                "class": {"value_type": "bool"},
                "style": {"value_type": "str"},
            },
        )

    annotated_cls = _annotation_refinable_class(annotation)
    if annotated_cls is not None:
        return Refinable(name=name, kind="class_ref", refinable_type=rtype, target=annotated_cls)

    if isinstance(default_value, type):
        return Refinable(
            name=name, kind="class_ref", refinable_type=rtype, target=_qual(default_value)
        )

    if isinstance(default_value, Namespace):
        d = dict(default_value)
        if d:
            return Refinable(
                name=name, kind="namespace", refinable_type=rtype, known_keys=sorted(d)
            )
        return Refinable(name=name, kind="open_namespace", refinable_type=rtype)

    if isinstance(refinable_obj, EvaluatedRefinable):
        return Refinable(name=name, kind="evaluated_scalar", refinable_type=rtype)

    return Refinable(name=name, kind="scalar", refinable_type=rtype)


def _collect_init_members(cls: type) -> list[str]:
    """Public ``self.X = …`` (and ``self.X: T = …``) names assigned in
    *cls*'s and its parents' ``__init__`` methods.

    Used to surface the configurable surface of "traditional" iommi
    classes — Cell/CellConfig and friends — that don't declare their API
    via ``Refinable()`` attributes. We walk ``cls.__mro__`` and AST-parse
    each class's own ``__init__`` source so the result captures attrs
    set anywhere in the inheritance chain. Decorator wrappers
    (e.g. ``@dispatch``) are unwrapped via ``__wrapped__`` when present.
    """
    names: set[str] = set()
    for base in cls.__mro__:
        if base is object:
            continue
        init = base.__dict__.get("__init__")
        if init is None:
            continue
        target = inspect.unwrap(init) if callable(init) else init
        try:
            src = inspect.getsource(target)
        except (OSError, TypeError):
            continue
        try:
            tree = ast.parse(textwrap.dedent(src))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            targets: list[ast.AST]
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            for t in targets:
                if (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                    and not t.attr.startswith("_")
                ):
                    names.add(t.attr)
    return sorted(names)


def _apply_traditional_overrides(
    cls_qualname: str, refinables: dict[str, Refinable]
) -> None:
    """Rewrite specific refinables to ``traditional_class`` kind in place.

    iommi declares ``Column.cell``/``Table.cell`` as ``Namespace``/scalar
    refinables, but at runtime they're passed as kwargs into a
    ``Cell.__init__`` chain. Static reflection can't see that
    relationship, so we patch the classification here.
    """
    overrides = _TRADITIONAL_TARGETS.get(cls_qualname)
    if not overrides:
        return
    for name, target in overrides.items():
        existing = refinables.get(name)
        rtype = existing.refinable_type if existing is not None else ""
        refinables[name] = Refinable(
            name=name,
            kind="traditional_class",
            refinable_type=rtype,
            target=target,
        )


def _reflect_class(cls: type) -> IommiClass:
    from iommi.refinable import Refinable as IommiRefinable

    meta = dict(cls.get_meta()) if hasattr(cls, "get_meta") else {}

    annotations: dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        annotations.update(getattr(base, "__annotations__", {}))

    refinables: dict[str, Refinable] = {}
    for n in sorted(dir(cls)):
        v = getattr(cls, n, None)
        if isinstance(v, IommiRefinable):
            refinables[n] = _classify(n, v, meta.get(n), annotations.get(n))

    qualname = _qual(cls)
    _apply_traditional_overrides(qualname, refinables)

    return IommiClass(
        qualname=qualname,
        bases=[_qual(b) for b in cls.__mro__[1:] if b is not object],
        refinables=refinables,
        init_members=_collect_init_members(cls),
    )


def build(seeds: list[str] | tuple[str, ...] = DEFAULT_SEEDS) -> IommiGraph:
    """BFS from *seeds*, reflecting each reachable class. Returns the graph."""
    bootstrap_django()

    import iommi

    seen: dict[str, IommiClass] = {}
    queue: list[str] = []
    skipped: list[str] = []

    for spec in seeds:
        cls = _import_seed(spec)
        if cls is None:
            skipped.append(spec)
            continue
        queue.append(_qual(cls))
        # Cache by qualname so duplicates in seeds collapse.
        if _qual(cls) not in seen:
            seen[_qual(cls)] = _reflect_class(cls)

    # BFS over class_ref / traditional_class targets and members member_classes.
    while queue:
        next_queue: list[str] = []
        for q in queue:
            ic = seen[q]
            for r in ic.refinables.values():
                if r.kind in ("class_ref", "traditional_class"):
                    follow = r.target
                elif r.kind == "members":
                    follow = r.member_class
                else:
                    follow = None
                if not follow or follow in seen:
                    continue
                cls = _import_seed(follow)
                if cls is None:
                    continue
                seen[follow] = _reflect_class(cls)
                next_queue.append(follow)
        queue = next_queue

    iommi_version: str | None = getattr(iommi, "__version__", None)
    if iommi_version is None:
        try:
            from importlib.metadata import PackageNotFoundError, version
            iommi_version = version("iommi")
        except (ImportError, PackageNotFoundError):
            iommi_version = None

    return IommiGraph(classes=seen, iommi_version=iommi_version)


def main(argv: list[str] | None = None) -> int:
    """``python -m iommi_lsp.analyzers.iommi.reflect [--seeds a,b,c]``.

    Used internally by ``iommi_lsp graph build`` when running the
    reflector in a subprocess (so the user's venv supplies iommi).
    Output is the graph JSON on stdout.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    seeds: tuple[str, ...] = DEFAULT_SEEDS
    if argv and argv[0] == "--seeds" and len(argv) > 1:
        seeds = tuple(s.strip() for s in argv[1].split(",") if s.strip())
    graph = build(seeds=seeds)
    from .graph import to_json

    sys.stdout.write(to_json(graph))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
