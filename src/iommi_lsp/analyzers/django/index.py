"""AST-only workspace index of Django models.

Purely static — we never import the user's code. That trades some
fidelity (string-based ``ForeignKey("app.Model")`` references that span
unusual module layouts can resolve incorrectly) for the property that
indexing never raises on a misconfigured project.

What we extract:

* every class that transitively inherits ``django.db.models.Model``;
* its declared concrete fields (name + field type);
* whether an explicit primary key is set (so we know if ``id`` is
  injected);
* ``Meta.abstract`` (used to skip table-bound assertions);
* the reverse-relation graph: for each FK / OneToOne / M2M target,
  the set of attribute names accessible on the *target* model
  (``related_name=`` if given, else ``<lowermodel>_set``).
"""

from __future__ import annotations

import ast
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ... import log
from .builtins import BUILTIN_MODULES
from .magic import DATE_FIELD_NAMES, FK_LIKE_FIELD_NAMES, RELATION_FIELD_NAMES


_log = log.get("django.index")


@dataclass
class FieldInfo:
    name: str
    field_type: str            # e.g. "CharField", "ForeignKey"
    target: str | None = None  # for relation fields: raw target ref
    related_name: str | None = None
    is_pk: bool = False        # explicit primary_key=True
    has_choices: bool = False  # any `choices=` kwarg present → get_FOO_display method
    # Source location of the field's *name* token (the LHS of the
    # ``name = Field(...)`` assignment) — used to back jump-to-definition
    # for reverse-relation accessors. Zero values mean "unknown" (set for
    # fields propagated through abstract bases or sourced from builtin
    # stubs, where the AST source isn't a single concrete line).
    defined_at_line: int = 0   # 1-based, matching ast.lineno
    defined_at_col: int = 0    # 0-based, matching ast.col_offset


@dataclass
class ModelInfo:
    qualname: str              # e.g. "myapp.models.User"
    module: str                # e.g. "myapp.models"
    name: str                  # e.g. "User"
    file_path: Path
    bases: list[str]           # raw resolved base names (for diagnostics)
    fields: dict[str, FieldInfo] = field(default_factory=dict)
    abstract: bool = False
    has_explicit_pk: bool = False
    is_builtin: bool = False   # injected from the contrib stub, not workspace

    @property
    def implicit_id(self) -> bool:
        return not self.has_explicit_pk and not self.abstract

    @property
    def pk_name(self) -> str:
        """Name of the actual primary-key field on this model.

        Returns the explicit ``primary_key=True`` field's name when the
        model declares one, otherwise ``"id"`` (Django's default).
        """
        for f in self.fields.values():
            if f.is_pk:
                return f.name
        return "id"

    @property
    def fk_id_accessors(self) -> set[str]:
        """``<field>_id`` accessors injected by ForeignKey/OneToOneField."""
        return {
            f"{f.name}_id"
            for f in self.fields.values()
            if f.field_type in FK_LIKE_FIELD_NAMES
        }

    @property
    def field_names(self) -> set[str]:
        return set(self.fields.keys())

    @property
    def generated_method_names(self) -> set[str]:
        """Names of methods Django generates for this model.

        * ``get_<field>_display`` for every field declared with
          ``choices=`` — abstract bases included (they're propagated
          into concrete subclasses by :func:`_propagate_inherited_fields`).
        * ``get_next_by_<field>`` / ``get_previous_by_<field>`` for every
          date/datetime field declared on a concrete model.

        Returned without parentheses — callers compare against attribute
        names from diagnostics, which never carry the call suffix.
        """
        out: set[str] = set()
        for f in self.fields.values():
            if f.has_choices:
                out.add(f"get_{f.name}_display")
            if f.field_type in DATE_FIELD_NAMES:
                out.add(f"get_next_by_{f.name}")
                out.add(f"get_previous_by_{f.name}")
        return out


@dataclass
class DjangoIndex:
    models: dict[str, ModelInfo] = field(default_factory=dict)
    # target model qualname -> {reverse attr name -> source model qualname}
    # The source is the model that *declared* the FK/M2M whose reverse
    # accessor lives on the target. Walkers need it to step through a
    # reverse relation and continue field validation on the source model.
    reverse_relations: dict[str, dict[str, str]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    # simple-name index for fast receiver lookup (e.g. "User" -> [qualname, ...])
    by_simple_name: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    # Union of method names declared on any class inheriting (transitively)
    # from ``django.db.models.QuerySet`` or ``Manager`` in the workspace.
    # Used to suppress ``Model.objects.<custom_method>`` false positives —
    # ty can't see methods added by ``objects = MyQuerySet.as_manager()``
    # without runtime stubs, so we accept any such method name as valid on
    # any manager attribute access. Bias is toward false negatives, in
    # line with the rest of the analyzer.
    custom_queryset_methods: set[str] = field(default_factory=set)
    # class qualname -> concrete model qualnames that transitively inherit
    # it. Keyed by *any* class in the base graph, including non-model
    # mixins (e.g. a plain ``class DuplicateSupport:`` mixed into models),
    # so ``self`` inside a mixin method can be resolved to the models that
    # actually use it. Excludes the class itself.
    descendants_of: dict[str, frozenset[str]] = field(default_factory=dict)

    def add_model(self, info: ModelInfo) -> None:
        self.models[info.qualname] = info
        self.by_simple_name[info.name].append(info.qualname)

    def reverse_attrs(self, model_qualname: str) -> set[str]:
        return set(self.reverse_relations.get(model_qualname, {}).keys())

    def reverse_source(self, model_qualname: str, reverse_name: str) -> str | None:
        """Source model qualname for a reverse accessor on *model_qualname*."""
        return self.reverse_relations.get(model_qualname, {}).get(reverse_name)

    def reverse_field(
        self, target_qualname: str, reverse_name: str,
    ) -> tuple[ModelInfo, FieldInfo] | None:
        """Find the source ``FieldInfo`` behind a reverse accessor.

        Given ``x.foos`` where ``x`` is a ``Foo`` instance, returns the
        ``(source_model, field)`` pair for the FK / M2M / O2O field whose
        ``related_name`` is ``"foos"`` (or whose default ``<lower>_set``
        matches). Used to back jump-to-definition: the ``field``'s
        ``defined_at_line`` / ``defined_at_col`` point at the declaration
        site in the source model.
        """
        source_qualname = self.reverse_source(target_qualname, reverse_name)
        if source_qualname is None:
            return None
        source_model = self.models.get(source_qualname)
        if source_model is None:
            return None
        for fi in source_model.fields.values():
            if fi.field_type not in RELATION_FIELD_NAMES:
                continue
            if fi.target != target_qualname:
                # Self-referential FKs use the source's own qualname as
                # target; reverse_source already filtered, so a mismatch
                # here means a different relation.
                continue
            name = fi.related_name or f"{source_model.name.lower()}_set"
            if name == reverse_name:
                return source_model, fi
        return None

    def lookup(self, simple_name: str) -> ModelInfo | None:
        """Return a model by simple class name; None if ambiguous or absent.

        When the workspace defines a model that shares its simple name with
        a builtin (e.g. a project's own ``User`` next to
        ``django.contrib.auth.models.User``), the workspace one wins. This
        keeps the contrib stub from poisoning name resolution on projects
        that swap out ``AUTH_USER_MODEL``.
        """
        candidates = self.by_simple_name.get(simple_name) or []
        if len(candidates) == 1:
            return self.models[candidates[0]]
        if len(candidates) > 1:
            workspace = [
                q for q in candidates if not self.models[q].is_builtin
            ]
            if len(workspace) == 1:
                return self.models[workspace[0]]
        return None

    def summary(self) -> str:
        lines = [f"DjangoIndex: {len(self.models)} models"]
        for qualname in sorted(self.models):
            m = self.models[qualname]
            tag = " [abstract]" if m.abstract else ""
            lines.append(f"  - {qualname}{tag}  ({len(m.fields)} fields)")
            for fname in sorted(m.fields):
                fi = m.fields[fname]
                detail = ""
                if fi.target:
                    detail = f" -> {fi.target}"
                    if fi.related_name:
                        detail += f"  related_name={fi.related_name!r}"
                lines.append(f"      {fname}: {fi.field_type}{detail}")
            rev = sorted((self.reverse_relations.get(qualname) or {}).keys())
            if rev:
                lines.append(f"      reverse: {', '.join(rev)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-qualname computation
# ---------------------------------------------------------------------------


def _module_qualname(workspace_root: Path, file_path: Path) -> str | None:
    """Best-effort dotted module name for *file_path* under *workspace_root*.

    We treat each top-level directory containing an ``__init__.py`` as a
    package root. ``__init__.py`` becomes the package itself (no trailing
    component).
    """
    try:
        rel = file_path.resolve().relative_to(workspace_root.resolve())
    except ValueError:
        return None
    parts = list(rel.parts)
    if not parts or not parts[-1].endswith(".py"):
        return None
    parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts.pop()
    if not parts:
        return None
    return ".".join(parts)


# ---------------------------------------------------------------------------
# AST walking
# ---------------------------------------------------------------------------


@dataclass
class _RawClass:
    """Per-file scrape of a class definition prior to model classification."""

    file_path: Path
    module: str
    name: str
    qualname: str
    base_strs: list[str]   # raw "Model", "models.Model", "MyAbstract", ...
    resolved_bases: list[str]  # bases mapped through file imports
    node: ast.ClassDef


@dataclass
class _FileScrape:
    module: str
    file_path: Path
    classes: list[_RawClass]
    # local name -> resolved qualified name (best effort)
    imports: dict[str, str]


def _flatten_attribute(node: ast.AST) -> str | None:
    """Render `a.b.c` AST into the dotted string ``"a.b.c"``; None otherwise."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _resolve_base(base_str: str, imports: dict[str, str]) -> str:
    """Map a raw base string like ``models.Model`` through the file's imports."""
    head, _, tail = base_str.partition(".")
    if head in imports:
        full = imports[head]
        return f"{full}.{tail}" if tail else full
    return base_str


def _resolve_relative_module(
    module: str | None, level: int, submodule: str | None, *, is_package: bool
) -> str | None:
    """Absolute qualname an ``from <dots><submodule> import …`` points at.

    *module* is the qualname of the importing file (``a.b.mod`` or, for a
    package ``__init__``, ``a.b``). Python anchors relative imports at the
    file's *package*: the module itself when it's a package, else its
    parent. ``level`` 1 is that package, each extra level climbs one more.
    Returns ``None`` when the import reaches above the top-level package or
    the module qualname is unknown.
    """
    if not module:
        return None
    parts = module.split(".")
    pkg_parts = parts if is_package else parts[:-1]
    up = level - 1
    if up > len(pkg_parts):
        return None
    base_parts = pkg_parts[: len(pkg_parts) - up]
    pieces = base_parts + ([submodule] if submodule else [])
    if not pieces:
        return None
    return ".".join(pieces)


def _build_import_map(
    tree: ast.Module, module: str | None, *, is_package: bool
) -> dict[str, str]:
    """Map each name a file binds via import to its source qualname.

    ``from a.b import C`` → ``{'C': 'a.b.C'}`` (honouring ``as`` aliases),
    ``import a.b`` → ``{'a': 'a'}``, and relative imports
    (``from .models import C``) resolved against *module*'s package.
    """
    imports: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                resolved = _resolve_relative_module(
                    module, node.level, node.module, is_package=is_package
                )
            else:
                resolved = node.module
            if not resolved:
                continue
            for alias in node.names:
                imports[alias.asname or alias.name] = f"{resolved}.{alias.name}"
    return imports


def _scrape_file(
    path: Path, module: str, *, source: str | None = None
) -> _FileScrape | None:
    if source is None:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            _log.debug("skipping %s: %s", path, e)
            return None
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        _log.debug("skipping %s (syntax error): %s", path, e)
        return None

    imports = _build_import_map(
        tree, module, is_package=path.name == "__init__.py"
    )

    classes: list[_RawClass] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        base_strs: list[str] = []
        for base in node.bases:
            flat = _flatten_attribute(base)
            if flat is None and isinstance(base, ast.Name):
                flat = base.id
            if flat is not None:
                base_strs.append(flat)
        resolved = [_resolve_base(b, imports) for b in base_strs]
        qualname = f"{module}.{node.name}"
        classes.append(
            _RawClass(
                file_path=path,
                module=module,
                name=node.name,
                qualname=qualname,
                base_strs=base_strs,
                resolved_bases=resolved,
                node=node,
            )
        )

    return _FileScrape(module=module, file_path=path, classes=classes, imports=imports)


# ---------------------------------------------------------------------------
# Model classification
# ---------------------------------------------------------------------------


_DJANGO_MODEL_BASE_FORMS = frozenset({
    "django.db.models.Model",
    # Resolved forms via `from django.db import models`:
    "django.db.models.Model",  # already, but covers `models.Model` post-resolve
})


def _looks_like_django_model_base(resolved_base: str) -> bool:
    return resolved_base == "django.db.models.Model"


def _classify_models(raws: list[_RawClass]) -> dict[str, _RawClass]:
    """Return the subset of *raws* that are (transitively) Django models.

    Iterates to fixed point: a class is a model iff it has a base that
    is the canonical Django Model OR matches another classified model
    (by qualname or simple name).
    """
    qualname_to_raw: dict[str, _RawClass] = {r.qualname: r for r in raws}
    by_simple: dict[str, list[_RawClass]] = defaultdict(list)
    for r in raws:
        by_simple[r.name].append(r)

    is_model: dict[str, bool] = {r.qualname: False for r in raws}

    changed = True
    while changed:
        changed = False
        for r in raws:
            if is_model[r.qualname]:
                continue
            for base in r.resolved_bases:
                if _looks_like_django_model_base(base):
                    is_model[r.qualname] = True
                    changed = True
                    break
                # Cross-module base by qualname (when imports resolved fully).
                if base in qualname_to_raw and is_model.get(base):
                    is_model[r.qualname] = True
                    changed = True
                    break
                # Same-file or unresolved bare-name base. Walk through all
                # classes in the project with that simple name. Restricted
                # to *bare* bases (no dot): a dotted base like
                # ``iommi.Action`` names a specific external class, and
                # matching it to a workspace model purely by the colliding
                # tail (``Action``) misclassifies every ``class Foo(lib.Foo)``
                # whose name happens to shadow a real model. Properly-imported
                # workspace bases resolve to a full qualname and are caught by
                # the exact-qualname check above.
                if "." in base:
                    continue
                same_name_candidates = by_simple.get(base, ())
                if any(is_model.get(c.qualname) for c in same_name_candidates):
                    is_model[r.qualname] = True
                    changed = True
                    break

    return {q: qualname_to_raw[q] for q, v in is_model.items() if v}


# ---------------------------------------------------------------------------
# Field extraction from a classified model body
# ---------------------------------------------------------------------------


def _string_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _bool_value(node: ast.AST) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _call_field_name(
    call: ast.Call,
    file_imports: dict[str, str],
    field_aliases: dict[str, str] | None = None,
) -> str | None:
    """If *call* looks like a Django field constructor, return its short name.

    Handles ``models.CharField(...)``, ``CharField(...)`` (when imported),
    and fully-qualified ``django.db.models.CharField(...)``.

    When *field_aliases* is supplied, a constructor whose class subclasses
    one of Django's relation field types (e.g. ``MyCustomFK(models.ForeignKey)``)
    is rewritten to the canonical parent name so downstream code that
    checks ``field_type in RELATION_FIELD_NAMES`` still matches.
    """
    flat = _flatten_attribute(call.func)
    if flat is None:
        if isinstance(call.func, ast.Name):
            flat = call.func.id
        else:
            return None
    # Resolve via imports.
    head, _, tail = flat.partition(".")
    if head in file_imports:
        full = file_imports[head]
        flat = f"{full}.{tail}" if tail else full
    if field_aliases:
        alias = field_aliases.get(flat)
        if alias is not None:
            return alias
    # Accept anything from django.db.models.* and any unqualified name —
    # we can't be 100% sure about unqualified, but for filtering purposes
    # over-matching is harmless (we'd record extra "fields" on a non-model
    # class, but that class wouldn't be in the model index in the first
    # place).
    last = flat.rsplit(".", 1)[-1]
    if field_aliases:
        alias = field_aliases.get(last)
        if alias is not None:
            return alias
    return last


def _extract_meta(class_node: ast.ClassDef) -> dict[str, ast.AST]:
    for stmt in class_node.body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == "Meta":
            attrs: dict[str, ast.AST] = {}
            for sub in stmt.body:
                if isinstance(sub, ast.Assign):
                    for tgt in sub.targets:
                        if isinstance(tgt, ast.Name):
                            attrs[tgt.id] = sub.value
                elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                    if sub.value is not None:
                        attrs[sub.target.id] = sub.value
            return attrs
    return {}


def _resolve_fk_target(
    arg: ast.AST,
    *,
    self_qualname: str,
    file_imports: dict[str, str],
    index: DjangoIndex,
) -> str | None:
    """Best-effort resolution of a relation field's first positional arg.

    Uses :meth:`DjangoIndex.lookup` for simple-name matches so a
    workspace model shadows a same-named builtin (the same rule that
    governs receiver resolution at scan time).
    """
    def _pick(simple: str) -> str | None:
        # A bare reference resolves to a class defined in the *same module*
        # if one exists there — Python scoping guarantees the local class
        # wins. This disambiguates simple names shared across apps (two
        # ``Service`` models in different apps) where ``lookup`` would tie
        # and give up.
        module = self_qualname.rsplit(".", 1)[0]
        same_module = f"{module}.{simple}"
        if same_module in index.models:
            return same_module
        info = index.lookup(simple)
        return info.qualname if info is not None else None

    s = _string_value(arg)
    if s is not None:
        if s == "self":
            return self_qualname
        if "." in s:
            # Django "app_label.ModelName" form — we don't track apps, so
            # match by simple name.
            simple = s.rsplit(".", 1)[-1]
        else:
            simple = s
        return _pick(simple)
    # Bare Name — `User`, possibly imported.
    if isinstance(arg, ast.Name):
        local = arg.id
        if local in file_imports:
            return file_imports[local]
        return _pick(local)
    # Attribute — `myapp.models.User`.
    flat = _flatten_attribute(arg)
    if flat is not None:
        head, _, tail = flat.partition(".")
        if head in file_imports:
            return f"{file_imports[head]}.{tail}" if tail else file_imports[head]
        return flat
    return None


def _populate_fields(
    info: ModelInfo,
    class_node: ast.ClassDef,
    file_imports: dict[str, str],
    index: DjangoIndex,
    field_aliases: dict[str, str] | None = None,
) -> None:
    for stmt in class_node.body:
        # Only top-level `name = Field(...)` style declarations.
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            continue
        if not isinstance(stmt.value, ast.Call):
            continue
        target_name_node = stmt.targets[0]
        field_name = target_name_node.id
        ftype = _call_field_name(stmt.value, file_imports, field_aliases)
        if ftype is None:
            continue
        fi = FieldInfo(
            name=field_name,
            field_type=ftype,
            defined_at_line=target_name_node.lineno,
            defined_at_col=target_name_node.col_offset,
        )
        # Inspect kwargs.
        for kw in stmt.value.keywords:
            if kw.arg == "primary_key" and _bool_value(kw.value):
                fi.is_pk = True
                info.has_explicit_pk = True
            elif kw.arg == "related_name":
                rn = _string_value(kw.value)
                if rn is not None:
                    fi.related_name = rn
            elif kw.arg == "choices":
                # Any non-None ``choices=`` value triggers Django's
                # ``get_<field>_display()`` method generation. We don't
                # validate the structure — even a variable reference
                # (``choices=STATUS_CHOICES``) is enough at runtime.
                fi.has_choices = True
        # Resolve relation target if applicable.
        if ftype in RELATION_FIELD_NAMES and stmt.value.args:
            fi.target = _resolve_fk_target(
                stmt.value.args[0],
                self_qualname=info.qualname,
                file_imports=file_imports,
                index=index,
            )
        info.fields[field_name] = fi

    meta_attrs = _extract_meta(class_node)
    abstract_node = meta_attrs.get("abstract")
    if abstract_node is not None and _bool_value(abstract_node) is True:
        info.abstract = True


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


_DEFAULT_SKIP = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "build", "dist", "site-packages",
})


# Synthetic anchor for builtin-stub file paths. Nothing reads these paths
# off disk; they exist so each builtin scrape has a unique, hashable key.
_BUILTIN_ROOT = Path("/__iommi_lsp_builtins__")


def _builtin_path(module: str) -> Path:
    parts = module.split(".")
    parts[-1] = parts[-1] + ".py"
    return _BUILTIN_ROOT.joinpath(*parts)


def _is_builtin_path(path: Path) -> bool:
    try:
        return path.is_relative_to(_BUILTIN_ROOT)
    except ValueError:
        return False


def _build_builtin_scrapes() -> dict[Path, _FileScrape]:
    """Parse each entry in :data:`BUILTIN_MODULES` into a scrape.

    Cached at module level wouldn't help much (each scrape is ~µs) and
    would complicate test isolation. Keep it simple: rebuild on each
    ``assemble_index`` call.
    """
    out: dict[Path, _FileScrape] = {}
    for module, source in BUILTIN_MODULES.items():
        path = _builtin_path(module)
        scrape = _scrape_file(path, module, source=source)
        if scrape is not None:
            out[path] = scrape
    return out


def _iter_python_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _DEFAULT_SKIP and not d.startswith(".")]
        for name in filenames:
            if name.endswith(".py"):
                found.append(Path(dirpath) / name)
    return found


def collect_scrapes(workspace_root: Path) -> dict[Path, _FileScrape]:
    """Walk *workspace_root* and parse every .py file once. Returns a
    cache keyed by absolute path so incremental updates can re-use
    every entry except the one that changed."""
    workspace_root = workspace_root.resolve()
    out: dict[Path, _FileScrape] = {}
    for py in _iter_python_files(workspace_root):
        scrape = scrape_file(workspace_root, py)
        if scrape is not None:
            out[py.resolve()] = scrape
    return out


def scrape_file(workspace_root: Path, file_path: Path) -> _FileScrape | None:
    """Parse a single file under *workspace_root*. Returns ``None`` if
    the file cannot be assigned a module qualname or fails to parse."""
    workspace_root = workspace_root.resolve()
    file_path = file_path.resolve()
    module = _module_qualname(workspace_root, file_path)
    if module is None:
        return None
    return _scrape_file(file_path, module)


def assemble_index(
    workspace_root: Path,
    scrapes: dict[Path, _FileScrape],
    *,
    include_builtins: bool = True,
) -> DjangoIndex:
    """Run classification, field extraction, and reverse-relation
    computation over a precomputed scrape map. Pure CPU work.

    When *include_builtins* is true (the default), a static stub of
    ``django.contrib.auth`` / ``contenttypes`` / ``sessions`` models is
    folded in so projects that use Django's built-in ``User`` get ORM-
    lookup validation without us having to import site-packages. A
    workspace model with the same simple name as a builtin (e.g. a
    project's own ``User``) still wins resolution.
    """
    workspace_root = workspace_root.resolve()
    all_scrapes: dict[Path, _FileScrape] = dict(scrapes)
    if include_builtins:
        for path, scrape in _build_builtin_scrapes().items():
            all_scrapes.setdefault(path, scrape)

    raws: list[_RawClass] = []
    file_imports: dict[Path, dict[str, str]] = {}
    for path, scrape in all_scrapes.items():
        raws.extend(scrape.classes)
        file_imports[path] = scrape.imports

    model_raws = _classify_models(raws)
    field_aliases = _classify_field_aliases(raws)
    index = DjangoIndex()

    # First pass: instantiate ModelInfo so cross-references resolve.
    for raw in model_raws.values():
        info = ModelInfo(
            qualname=raw.qualname,
            module=raw.module,
            name=raw.name,
            file_path=raw.file_path,
            bases=list(raw.resolved_bases),
            is_builtin=_is_builtin_path(raw.file_path),
        )
        index.add_model(info)

    # Second pass: populate fields (now `index.by_simple_name` is full).
    for raw in model_raws.values():
        info = index.models[raw.qualname]
        _populate_fields(
            info,
            raw.node,
            file_imports.get(raw.file_path, {}),
            index,
            field_aliases,
        )

    # Third pass: propagate inherited fields. Django's metaclass copies
    # abstract-base fields into the concrete subclass; concrete-base
    # fields are queryable on the subclass too (multi-table JOIN). We
    # don't distinguish — for kwarg validation, "the subclass knows about
    # this field" is what matters. Iterate to fixed point so chains
    # (User -> AbstractUser -> AbstractBaseUser) resolve regardless of
    # which order we walk.
    _propagate_inherited_fields(model_raws, index)

    # Fourth pass: build reverse_relations. Only concrete models register
    # reverse accessors — abstract bases that declare e.g. an M2M with a
    # related_name share the FieldInfo across every concrete subclass
    # after propagation, and Django itself requires ``%(class)s`` in
    # related_name when an abstract base owns a relation. Walking
    # concrete models only means we record one reverse per concrete
    # subclass, which matches runtime.
    for info in index.models.values():
        if info.abstract:
            continue
        for fi in info.fields.values():
            if fi.field_type not in RELATION_FIELD_NAMES:
                continue
            target = fi.target
            if target is None or target not in index.models:
                # We still want self-referential FKs even if they used a string.
                if target == info.qualname and fi.target == info.qualname:
                    pass
                else:
                    continue
            reverse_name = fi.related_name
            if reverse_name is None:
                reverse_name = f"{info.name.lower()}_set"
            if reverse_name == "+":
                # Django convention: disables the reverse relation.
                continue
            index.reverse_relations[target][reverse_name] = info.qualname

    # Fifth pass: workspace-wide union of custom QuerySet / Manager method
    # names. ``objects = MyQuerySet.as_manager()`` exposes those methods on
    # the manager — ty doesn't see them. We don't try to link a specific
    # model to a specific QuerySet (would need flow analysis across files);
    # the union is enough to suppress false positives, biased FN as usual.
    index.custom_queryset_methods.update(
        _collect_queryset_method_names(raws)
    )

    # Sixth pass: map every class in the base graph to the concrete models
    # that inherit it, so ``self`` inside a non-model mixin resolves to its
    # concrete model users.
    index.descendants_of = _build_descendant_map(raws, index.models)

    return index


def _build_descendant_map(
    raws: list[_RawClass], models: dict[str, ModelInfo]
) -> dict[str, frozenset[str]]:
    """For each class qualname, the concrete (non-abstract) model qualnames
    that transitively inherit from it.

    Walks the full base graph across model *and* non-model classes — the
    point is to reach plain mixins that never get classified as models — and
    resolves bases by exact qualname first, then by simple name (same-file
    or unresolved references), mirroring :func:`_classify_models`.
    """
    by_qual = {r.qualname: r for r in raws}
    by_simple: dict[str, list[_RawClass]] = defaultdict(list)
    for r in raws:
        by_simple[r.name].append(r)

    def ancestors(start: str) -> set[str]:
        seen: set[str] = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            r = by_qual.get(cur)
            if r is None:
                continue
            for base in r.resolved_bases:
                if base in by_qual:
                    cands = [base]
                else:
                    head = base.split(".")[-1]
                    cands = [c.qualname for c in by_simple.get(head, ())]
                for c in cands:
                    if c not in seen:
                        seen.add(c)
                        stack.append(c)
        seen.discard(start)
        return seen

    descendants: dict[str, set[str]] = defaultdict(set)
    for qualname, info in models.items():
        if info.abstract:
            continue
        for anc in ancestors(qualname):
            descendants[anc].add(qualname)
    return {k: frozenset(v) for k, v in descendants.items()}


_QUERYSET_BASES = frozenset({
    "django.db.models.QuerySet",
    "django.db.models.Manager",
    "django.db.models.manager.BaseManager",
    "django.db.models.manager.Manager",
})


# Canonical Django qualnames for the relation field types whose subclasses
# we want to recognise as their parent. ``OneToOneField`` itself extends
# ``ForeignKey`` at runtime, but we treat it as its own canonical name —
# matching the rest of the analyzer, which checks the literal string.
_RELATION_FIELD_QUALNAMES: dict[str, str] = {
    "django.db.models.ForeignKey": "ForeignKey",
    "django.db.models.OneToOneField": "OneToOneField",
    "django.db.models.ManyToManyField": "ManyToManyField",
}


def _classify_field_aliases(raws: list[_RawClass]) -> dict[str, str]:
    """Map workspace classes that subclass a Django relation field to the
    canonical parent's short name.

    Runtime Django treats ``class MyFK(models.ForeignKey): ...`` exactly
    like a ``ForeignKey`` for the purposes that matter here (reverse
    relations, ``<name>_id`` accessor, M2M.through). Our checks key off
    the literal ``field_type`` string, so resolve subclass names back to
    the parent before recording the field.

    Returned map covers both fully-qualified names (e.g.
    ``"myapp.fields.MyFK"``) and unambiguous simple names (e.g. ``"MyFK"``)
    so :func:`_call_field_name` can resolve regardless of how the field
    class is referenced in user code.
    """
    qualname_to_raw = {r.qualname: r for r in raws}
    by_simple: dict[str, list[_RawClass]] = defaultdict(list)
    for r in raws:
        by_simple[r.name].append(r)

    canonical: dict[str, str] = {}  # qualname -> canonical short name

    def _base_canonical(base: str) -> str | None:
        c = _RELATION_FIELD_QUALNAMES.get(base)
        if c is not None:
            return c
        c = canonical.get(base)
        if c is not None:
            return c
        head = base.rsplit(".", 1)[-1]
        for cls in by_simple.get(head, ()):
            c = canonical.get(cls.qualname)
            if c is not None:
                return c
        return None

    changed = True
    while changed:
        changed = False
        for r in raws:
            if r.qualname in canonical:
                continue
            for base in r.resolved_bases:
                c = _base_canonical(base)
                if c is not None:
                    canonical[r.qualname] = c
                    changed = True
                    break

    aliases: dict[str, str] = dict(_RELATION_FIELD_QUALNAMES)
    aliases.update(canonical)
    # Add unambiguous simple-name aliases so unresolved bare references
    # (``class MyFK(models.ForeignKey)`` then ``MyFK(...)`` in the same
    # module without an explicit import line) still get rewritten.
    simple_seen: dict[str, set[str]] = defaultdict(set)
    for qn, c in canonical.items():
        simple_seen[qn.rsplit(".", 1)[-1]].add(c)
    for simple, vals in simple_seen.items():
        if len(vals) == 1:
            aliases.setdefault(simple, next(iter(vals)))
    return aliases


def _collect_queryset_method_names(raws: list[_RawClass]) -> set[str]:
    """Methods declared on every workspace class that transitively inherits
    from ``QuerySet`` or ``Manager``.

    Builds a fixed-point classification (like :func:`_classify_models`):
    a class is a QuerySet/Manager iff one of its resolved bases is the
    canonical Django shape or another already-classified workspace class.
    """
    qualname_to_raw = {r.qualname: r for r in raws}
    by_simple: dict[str, list[_RawClass]] = defaultdict(list)
    for r in raws:
        by_simple[r.name].append(r)

    is_qs: dict[str, bool] = {r.qualname: False for r in raws}
    changed = True
    while changed:
        changed = False
        for r in raws:
            if is_qs[r.qualname]:
                continue
            for base in r.resolved_bases:
                if base in _QUERYSET_BASES:
                    is_qs[r.qualname] = True
                    changed = True
                    break
                if base in qualname_to_raw and is_qs.get(base):
                    is_qs[r.qualname] = True
                    changed = True
                    break
                head = base.split(".")[-1]
                if any(is_qs.get(c.qualname) for c in by_simple.get(head, ())):
                    is_qs[r.qualname] = True
                    changed = True
                    break

    out: set[str] = set()
    for r in raws:
        if not is_qs[r.qualname]:
            continue
        for stmt in r.node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not stmt.name.startswith("_"):
                    out.add(stmt.name)
    return out


def _propagate_inherited_fields(
    model_raws: dict[str, _RawClass], index: DjangoIndex
) -> None:
    """Copy fields from base ModelInfos into their subclasses.

    Each model's resolved bases are walked; bases that match an entry in
    *index.models* (by qualname or by simple-name with a unique
    candidate) contribute their fields to the subclass. Iterates to
    fixed point so multi-level chains converge regardless of walk
    order. A field's ``is_pk`` carries through to ``has_explicit_pk`` on
    the subclass.
    """
    changed = True
    while changed:
        changed = False
        for raw in model_raws.values():
            target = index.models[raw.qualname]
            for base_str in raw.resolved_bases:
                base_info = _resolve_base_to_model(base_str, index)
                if base_info is None or base_info.qualname == target.qualname:
                    continue
                for fname, fi in base_info.fields.items():
                    if fname in target.fields:
                        continue
                    target.fields[fname] = fi
                    if fi.is_pk:
                        target.has_explicit_pk = True
                    changed = True


def _resolve_base_to_model(
    base_str: str, index: DjangoIndex
) -> ModelInfo | None:
    if base_str in index.models:
        return index.models[base_str]
    simple = base_str.rsplit(".", 1)[-1]
    candidates = index.by_simple_name.get(simple) or []
    if len(candidates) == 1:
        return index.models[candidates[0]]
    return None


def build_index(workspace_root: Path) -> DjangoIndex:
    """AST-scan *workspace_root* for Django models. Pure, no I/O on user code."""
    return assemble_index(workspace_root, collect_scrapes(workspace_root))


def update_scrapes(
    workspace_root: Path,
    scrapes: dict[Path, _FileScrape],
    changed_path: Path,
) -> dict[Path, _FileScrape]:
    """Mutate *scrapes* in place to reflect a single file change.

    * If the file no longer exists or is no longer a Python file we can
      qualify, drop its entry.
    * Otherwise re-parse and replace the entry.

    Returns the same dict for chaining.
    """
    workspace_root = workspace_root.resolve()
    key = changed_path.resolve()
    if not key.exists() or not key.suffix == ".py":
        scrapes.pop(key, None)
        return scrapes
    new = scrape_file(workspace_root, key)
    if new is None:
        scrapes.pop(key, None)
    else:
        scrapes[key] = new
    return scrapes
