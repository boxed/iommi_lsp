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
from .magic import FK_LIKE_FIELD_NAMES, RELATION_FIELD_NAMES


_log = log.get("django.index")


@dataclass
class FieldInfo:
    name: str
    field_type: str            # e.g. "CharField", "ForeignKey"
    target: str | None = None  # for relation fields: raw target ref
    related_name: str | None = None
    is_pk: bool = False        # explicit primary_key=True


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

    @property
    def implicit_id(self) -> bool:
        return not self.has_explicit_pk and not self.abstract

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


@dataclass
class DjangoIndex:
    models: dict[str, ModelInfo] = field(default_factory=dict)
    # target model qualname -> set of reverse attr names
    reverse_relations: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # simple-name index for fast receiver lookup (e.g. "User" -> [qualname, ...])
    by_simple_name: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def add_model(self, info: ModelInfo) -> None:
        self.models[info.qualname] = info
        self.by_simple_name[info.name].append(info.qualname)

    def reverse_attrs(self, model_qualname: str) -> set[str]:
        return self.reverse_relations.get(model_qualname, set())

    def lookup(self, simple_name: str) -> ModelInfo | None:
        """Return a model by simple class name; None if ambiguous or absent."""
        candidates = self.by_simple_name.get(simple_name) or []
        if len(candidates) == 1:
            return self.models[candidates[0]]
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
            rev = sorted(self.reverse_relations.get(qualname, ()))
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


def _scrape_file(path: Path, module: str) -> _FileScrape | None:
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

    imports: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module is None or node.level:
                # Skip relative imports for v1 — tracking package context
                # would require building a package tree first.
                continue
            for alias in node.names:
                local = alias.asname or alias.name
                imports[local] = f"{node.module}.{alias.name}"

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
                # Same-file or unresolved-but-simple-name base. Walk through
                # all classes in the project with that simple name (head).
                head = base.split(".")[-1]
                same_name_candidates = by_simple.get(head, ())
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


def _call_field_name(call: ast.Call, file_imports: dict[str, str]) -> str | None:
    """If *call* looks like a Django field constructor, return its short name.

    Handles ``models.CharField(...)``, ``CharField(...)`` (when imported),
    and fully-qualified ``django.db.models.CharField(...)``.
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
    # Accept anything from django.db.models.* and any unqualified name —
    # we can't be 100% sure about unqualified, but for filtering purposes
    # over-matching is harmless (we'd record extra "fields" on a non-model
    # class, but that class wouldn't be in the model index in the first
    # place).
    last = flat.rsplit(".", 1)[-1]
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
    by_simple: dict[str, list[str]],
) -> str | None:
    """Best-effort resolution of a relation field's first positional arg."""
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
        candidates = by_simple.get(simple) or []
        if len(candidates) == 1:
            return candidates[0]
        return None
    # Bare Name — `User`, possibly imported.
    if isinstance(arg, ast.Name):
        local = arg.id
        if local in file_imports:
            return file_imports[local]
        candidates = by_simple.get(local) or []
        if len(candidates) == 1:
            return candidates[0]
        return None
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
    by_simple: dict[str, list[str]],
) -> None:
    for stmt in class_node.body:
        # Only top-level `name = Field(...)` style declarations.
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            continue
        if not isinstance(stmt.value, ast.Call):
            continue
        field_name = stmt.targets[0].id
        ftype = _call_field_name(stmt.value, file_imports)
        if ftype is None:
            continue
        fi = FieldInfo(name=field_name, field_type=ftype)
        # Inspect kwargs.
        for kw in stmt.value.keywords:
            if kw.arg == "primary_key" and _bool_value(kw.value):
                fi.is_pk = True
                info.has_explicit_pk = True
            elif kw.arg == "related_name":
                rn = _string_value(kw.value)
                if rn is not None:
                    fi.related_name = rn
        # Resolve relation target if applicable.
        if ftype in RELATION_FIELD_NAMES and stmt.value.args:
            fi.target = _resolve_fk_target(
                stmt.value.args[0],
                self_qualname=info.qualname,
                file_imports=file_imports,
                by_simple=by_simple,
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
    workspace_root: Path, scrapes: dict[Path, _FileScrape]
) -> DjangoIndex:
    """Run classification, field extraction, and reverse-relation
    computation over a precomputed scrape map. Pure CPU work."""
    workspace_root = workspace_root.resolve()
    raws: list[_RawClass] = []
    file_imports: dict[Path, dict[str, str]] = {}
    for path, scrape in scrapes.items():
        raws.extend(scrape.classes)
        file_imports[path] = scrape.imports

    model_raws = _classify_models(raws)
    index = DjangoIndex()

    # First pass: instantiate ModelInfo so cross-references resolve.
    for raw in model_raws.values():
        info = ModelInfo(
            qualname=raw.qualname,
            module=raw.module,
            name=raw.name,
            file_path=raw.file_path,
            bases=list(raw.resolved_bases),
        )
        index.add_model(info)

    # Second pass: populate fields (now `index.by_simple_name` is full).
    by_simple = index.by_simple_name
    for raw in model_raws.values():
        info = index.models[raw.qualname]
        _populate_fields(
            info,
            raw.node,
            file_imports.get(raw.file_path, {}),
            by_simple,
        )

    # Third pass: build reverse_relations.
    for info in index.models.values():
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
            index.reverse_relations[target].add(reverse_name)

    _log.info(
        "indexed %s: %d models, %d reverse relations",
        workspace_root,
        len(index.models),
        sum(len(v) for v in index.reverse_relations.values()),
    )
    return index


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
