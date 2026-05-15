"""Django analyzer: drops ``unresolved-attribute`` diagnostics whose
target is a Django metaclass-injected attribute on a recognised model.

For v1 we recognise the receiver type via two cheap heuristics
(``DESIGN.md`` §6.3):

* (a) **Syntactic match.** Bare ``Name`` whose simple identifier matches
  a known model class — e.g. ``User.objects``.
* (b) **Local flow.** Same-function assignments where the RHS is
  ``Model(...)`` or ``Model.objects.<query>(...)``  — e.g.
  ``user = User(...); user.pk``. Only the most recent assignment wins.

Anything outside these cases is forwarded unchanged. The bias is
explicitly toward false negatives: we'd rather leak some noise than
suppress a real bug.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from typing import TYPE_CHECKING

from ... import log
from ..base import Analyzer, CompletionResult, Diagnostic
from . import lookup_walker
from .index import (
    DjangoIndex,
    ModelInfo,
    _FileScrape,
    assemble_index,
    collect_scrapes,
    update_scrapes,
)
from .magic import FK_LIKE_FIELD_NAMES, ORM_LOOKUP_NAMES, RELATION_FIELD_NAMES

if TYPE_CHECKING:
    from ...config import Config


_log = log.get("django.analyzer")


_QUERY_METHODS_RETURNING_INSTANCE = frozenset({
    "get", "first", "last", "earliest", "latest",
    "create", "get_or_create", "update_or_create",
})

# QuerySet methods that return another QuerySet of the same model — i.e.
# iterating their result yields instances of the model. Used to resolve
# the bound variable in ``for p in Model.objects.filter(...)`` and the
# equivalent comprehension shape.
_QUERY_METHODS_RETURNING_QUERYSET = frozenset({
    "all", "filter", "exclude", "order_by", "reverse", "distinct",
    "none", "using", "select_related", "prefetch_related",
    "annotate", "alias", "defer", "only",
})

# Manager methods that take ``field__lookup=...`` kwargs we want to validate.
# ``update``/``create`` only accept single-segment field names in real
# Django — the walker is permissive about ``__`` traversal which means
# we'd miss e.g. ``update(author__name='x')`` (invalid Django). Bias FN.
_LOOKUP_METHODS = frozenset({
    "filter", "exclude", "get", "get_or_create", "update_or_create",
    "update", "create",
})

# Methods whose positional args are field-path strings (``order_by``-style).
# Each string is a chain like ``"author__name"`` (with optional leading
# ``-`` for ``order_by`` descending; ``"?"`` for random).
_FIELD_PATH_METHODS = frozenset({
    "order_by", "values", "values_list", "only", "defer", "distinct",
    "select_related", "prefetch_related",
})

# Methods whose kwarg *names* are user-defined aliases (not model fields)
# but whose *values* commonly contain ``F('path')`` / ``Count('path')``
# expressions — so we still want to walk the call to validate those.
_AGGREGATE_METHODS = frozenset({"annotate", "aggregate", "alias"})

_MANAGER_NAMES = frozenset({"objects", "_default_manager", "_base_manager"})

# Helper functions whose first positional arg is a model (class or queryset)
# and whose kwargs are ORM lookups (like ``filter(...)``).
# Django ships these under ``django.shortcuts``; we match by simple name.
_HELPER_LOOKUP_FUNCS = frozenset({
    "get_object_or_404", "get_list_or_404",
})

# Kwargs that some manager methods accept which are NOT field names
# (e.g. `defaults={...}` to ``get_or_create``). Skipping them keeps the
# scanner from raising spurious "unknown field" diagnostics.
_METHOD_ONLY_KWARGS = frozenset({"defaults", "create_defaults"})

_ORM_LOOKUP_DIAG_CODE = "django-unknown-orm-lookup"
_ORM_LOOKUP_DIAG_SOURCE = "iommi_lsp"


@dataclass
class _ParsedFile:
    tree: ast.Module
    source: str


class DjangoAnalyzer:
    """Implements the :class:`Analyzer` Protocol."""

    name = "django"

    def __init__(
        self,
        workspace_root: Path,
        django_index: DjangoIndex | None = None,
        config: "Config | None" = None,
        text_provider: Callable[[str], str | None] | None = None,
    ) -> None:
        # Lazy import — config.py pulls in this package via magic.py and we
        # need to break the cycle.
        from ...config import DEFAULT as DEFAULT_CONFIG

        self.workspace_root = workspace_root
        self.django_index: DjangoIndex = django_index or DjangoIndex()
        self.config: "Config" = config or DEFAULT_CONFIG
        self._text_provider = text_provider
        self._cache: dict[str, _ParsedFile] = {}
        self._scrapes: dict[Path, _FileScrape] = {}

    # -- Analyzer protocol ----------------------------------------------------

    async def index(self, workspace_root: Path) -> None:
        from ...config import load as load_config

        self.workspace_root = workspace_root
        self.config = load_config(workspace_root)
        self._scrapes = collect_scrapes(workspace_root)
        self.django_index = assemble_index(workspace_root, self._scrapes)
        self._cache.clear()
        _log.info(
            "indexed %s: %d models, %d reverse relations",
            workspace_root,
            len(self.django_index.models),
            sum(len(v) for v in self.django_index.reverse_relations.values()),
        )

    async def on_file_changed(self, uri: str) -> None:
        self._cache.pop(uri, None)
        path = _uri_to_path(uri)
        if path is None:
            return
        # Incremental: only re-parse the changed file, then re-run
        # classification + reverse-relation computation against the
        # cached scrape map. ~milliseconds even on large workspaces.
        key = path.resolve()
        had_classes = bool(self._scrapes.get(key) and self._scrapes[key].classes)
        update_scrapes(self.workspace_root, self._scrapes, path)
        has_classes = bool(self._scrapes.get(key) and self._scrapes[key].classes)
        # No class definitions before or after the edit → the index can't
        # have changed. Skip the reassembly to avoid burning CPU on every
        # keystroke in unrelated files.
        if not had_classes and not has_classes:
            return
        self.django_index = assemble_index(self.workspace_root, self._scrapes)

    # -- internals ------------------------------------------------------------

    def _evaluate(self, uri: str, diagnostic: Diagnostic) -> bool:
        path = _uri_to_path(uri)
        if path is None:
            return False
        parsed = self._parse(uri, path)
        if parsed is None:
            return False

        attr_node = _find_attribute_at(parsed.tree, diagnostic.get("range") or {})
        if attr_node is None:
            return False
        attr_name = attr_node.attr
        receiver = attr_node.value

        model = self._resolve_receiver_model(receiver, parsed.tree)
        if model is not None and self._attr_is_magic(model, attr_name):
            return True

        # Narrow fallback for instance-only Django magic on annotated
        # receivers: ty knows the receiver's type from an annotation
        # (parameter, ``self`` in a model method, annotated assignment)
        # but our flow-based resolver doesn't follow annotations. A few
        # kinds of attr are safe to suppress this way — the names are
        # registered in the index, so genuine typos still leak through:
        #   * ``<field>_id`` underlying-column accessors on FK fields;
        #   * reverse-relation accessors (``<lower>_set`` default and
        #     explicit ``related_name=``);
        #   * ``pk`` and the model's actual primary-key field name —
        #     Django adds ``pk`` to every instance regardless of the
        #     declared PK field's name, and descriptor magic can hide
        #     the real PK field from ty too.
        if (
            attr_name.endswith("_id")
            and self.config.is_rule_enabled("fk_id")
        ):
            ann_model = self._resolve_via_annotation(receiver, parsed.tree)
            if ann_model is not None and attr_name in ann_model.fk_id_accessors:
                return True

        if self.config.is_rule_enabled("reverse"):
            ann_model = self._resolve_via_annotation(receiver, parsed.tree)
            if (
                ann_model is not None
                and attr_name in self.django_index.reverse_attrs(ann_model.qualname)
            ):
                return True

        if self.config.is_rule_enabled("pk"):
            ann_model = self._resolve_via_annotation(receiver, parsed.tree)
            if ann_model is not None:
                # ``pk`` exists on every model instance regardless of the
                # declared PK name; ``ann_model.pk_name`` is ``id`` for
                # implicit-PK models or the explicit ``primary_key=True``
                # field's name otherwise.
                if attr_name == "pk" or attr_name == ann_model.pk_name:
                    return True

        # ``<m2m>.through`` — Django attaches ``through`` to every
        # ManyToManyField descriptor. ty can't see it without runtime
        # stubs, so suppress when the receiver is a known M2M field on a
        # recognised model.
        if attr_name == "through" and self._is_m2m_receiver(receiver, parsed.tree):
            return True

        # ``<Model>.<manager>.<custom_method>`` — custom QuerySet methods
        # surfaced via ``objects = MyQuerySet.as_manager()``. ty doesn't
        # see those methods. We resolve the manager attribute access and
        # check the workspace-wide queryset method union.
        if self._is_manager_method_access(receiver, attr_name, parsed.tree):
            return True

        return False

    def _is_manager_method_access(
        self, receiver: ast.AST, attr_name: str, tree: ast.Module,
    ) -> bool:
        """``<receiver_resolving_to_model>.<manager_name>.<attr_name>``
        where *attr_name* is in the workspace's QuerySet method union.
        """
        if attr_name not in self.django_index.custom_queryset_methods:
            return False
        if not isinstance(receiver, ast.Attribute):
            return False
        if receiver.attr not in _MANAGER_NAMES:
            return False
        owner = receiver.value
        # ``Model.objects.method`` — owner is a Name resolving to a model.
        if isinstance(owner, ast.Name):
            if self.django_index.lookup(owner.id) is not None:
                return True
            local = self._resolve_local_variable(owner.id, owner, tree)
            return local is not None
        # ``models.User.objects.method`` — dotted-path access.
        if isinstance(owner, ast.Attribute):
            return self.django_index.lookup(owner.attr) is not None
        return False

    def _is_m2m_receiver(self, receiver: ast.AST, tree: ast.Module) -> bool:
        """``receiver`` is ``<model_or_instance>.<m2m_field_name>``?"""
        if not isinstance(receiver, ast.Attribute):
            return False
        owner_model = self._resolve_receiver_model(receiver.value, tree)
        if owner_model is None:
            owner_model = self._resolve_via_annotation(receiver.value, tree)
        if owner_model is None:
            return False
        fi = owner_model.fields.get(receiver.attr)
        return fi is not None and fi.field_type == "ManyToManyField"

    def _parse(self, uri: str, path: Path) -> _ParsedFile | None:
        source = self._source_for(uri, path)
        if source is None:
            return None
        cached = self._cache.get(uri)
        if cached is not None and cached.source == source:
            return cached
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            _log.debug("could not parse %s: %s", path, e)
            return None
        parsed = _ParsedFile(tree=tree, source=source)
        self._cache[uri] = parsed
        return parsed

    def _source_for(self, uri: str, path: Path) -> str | None:
        if self._text_provider is not None:
            text = self._text_provider(uri)
            if text is not None:
                return text
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            _log.debug("could not read %s: %s", path, e)
            return None

    def _resolve_receiver_model(
        self, receiver: ast.AST, tree: ast.Module
    ) -> ModelInfo | None:
        # (a) Syntactic match: bare Name -> class lookup by simple name.
        if isinstance(receiver, ast.Name):
            model = self.django_index.lookup(receiver.id)
            if model is not None:
                return model
            # (b) Local flow: search enclosing scope for an assignment.
            return self._resolve_local_variable(receiver.id, receiver, tree)
        return None

    def _resolve_local_variable(
        self, var_name: str, use_site: ast.AST, tree: ast.Module
    ) -> ModelInfo | None:
        scope = _enclosing_function(tree, use_site)
        if scope is None:
            scope = tree
        use_pos = (getattr(use_site, "lineno", 0), getattr(use_site, "col_offset", 0))

        # Comprehensions have their own scope — a ``for p in qs`` in a
        # ListComp/DictComp/SetComp/GeneratorExp binds ``p`` only inside
        # that comprehension and shadows any outer binding. Check these
        # first so the shadow wins.
        for node in ast.walk(scope):
            if not isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                continue
            if not _node_contains(node, use_site):
                continue
            for gen in node.generators:
                if isinstance(gen.target, ast.Name) and gen.target.id == var_name:
                    inferred = self._infer_iter_yields_model(gen.iter)
                    if inferred is not None:
                        return inferred

        # Statement-level bindings: assignments and ``for`` loops.
        # Last match preceding the use site wins.
        last_match: ModelInfo | None = None
        for stmt in ast.walk(scope):
            if isinstance(stmt, ast.Assign):
                if (stmt.lineno, stmt.col_offset) >= use_pos:
                    continue
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == var_name:
                        inferred = self._infer_call_result_model(stmt.value)
                        if inferred is not None:
                            last_match = inferred
            elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                if (stmt.lineno, stmt.col_offset) >= use_pos:
                    continue
                if isinstance(stmt.target, ast.Name) and stmt.target.id == var_name:
                    inferred = self._infer_iter_yields_model(stmt.iter)
                    if inferred is not None:
                        last_match = inferred
        return last_match

    def _infer_iter_yields_model(self, value: ast.AST) -> ModelInfo | None:
        """Recognise iterables that yield Django model instances.

        Covers ``Model.objects`` (bare manager) and chained queryset
        methods like ``Model.objects.filter(...).order_by(...)``.
        """
        # Bare ``Model.objects`` / ``Model._default_manager`` — managers
        # are iterable and yield instances.
        if (
            isinstance(value, ast.Attribute)
            and value.attr in _MANAGER_NAMES
            and isinstance(value.value, ast.Name)
        ):
            return self.django_index.lookup(value.value.id)
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            method = value.func.attr
            if method in _QUERY_METHODS_RETURNING_QUERYSET:
                return self._infer_iter_yields_model(value.func.value)
        return None

    def _resolve_via_annotation(
        self, receiver: ast.AST, tree: ast.Module
    ) -> ModelInfo | None:
        """Resolve a receiver to a model via static type annotations.

        Covers three shapes the flow-based resolver misses:

        * ``self`` inside a method of a Django model class.
        * A function parameter with a model annotation (``def f(p: Profile)``).
        * An annotated assignment (``p: Profile = ...``).

        Returns ``None`` if the receiver isn't a bare name or no
        annotation resolves to a known model.
        """
        if not isinstance(receiver, ast.Name):
            return None
        var_name = receiver.id

        if var_name == "self":
            cls = _enclosing_class(tree, receiver)
            if cls is not None:
                model = self.django_index.lookup(cls.name)
                if model is not None:
                    return model

        scope = _enclosing_function(tree, receiver)
        if scope is not None and isinstance(
            scope, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            params = (
                list(scope.args.posonlyargs)
                + list(scope.args.args)
                + list(scope.args.kwonlyargs)
            )
            if scope.args.vararg is not None:
                params.append(scope.args.vararg)
            if scope.args.kwarg is not None:
                params.append(scope.args.kwarg)
            for arg in params:
                if arg.arg == var_name:
                    model = self._model_from_annotation(arg.annotation)
                    if model is not None:
                        return model
                    break

        search_scope: ast.AST = scope if scope is not None else tree
        use_pos = (
            getattr(receiver, "lineno", 0),
            getattr(receiver, "col_offset", 0),
        )
        last_match: ModelInfo | None = None
        for stmt in ast.walk(search_scope):
            if not isinstance(stmt, ast.AnnAssign):
                continue
            if (stmt.lineno, stmt.col_offset) >= use_pos:
                continue
            target = stmt.target
            if isinstance(target, ast.Name) and target.id == var_name:
                model = self._model_from_annotation(stmt.annotation)
                if model is not None:
                    last_match = model
        return last_match

    def _model_from_annotation(self, ann: ast.AST | None) -> ModelInfo | None:
        if ann is None:
            return None
        if isinstance(ann, ast.Name):
            return self.django_index.lookup(ann.id)
        if isinstance(ann, ast.Attribute):
            return self.django_index.lookup(ann.attr)
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            # Quoted/forward-ref: `"Profile"` or `"app.models.Profile"`.
            simple = ann.value.rsplit(".", 1)[-1]
            return self.django_index.lookup(simple)
        if isinstance(ann, ast.Subscript):
            # Unwrap `Optional[Profile]` / `list[Profile]` / `X | Y` slice.
            return self._model_from_annotation(ann.slice)
        if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
            # `Profile | None` — pick whichever side resolves.
            return (
                self._model_from_annotation(ann.left)
                or self._model_from_annotation(ann.right)
            )
        if isinstance(ann, ast.Tuple):
            for elt in ann.elts:
                model = self._model_from_annotation(elt)
                if model is not None:
                    return model
        return None

    def _infer_call_result_model(self, value: ast.AST) -> ModelInfo | None:
        """Recognise ``Model(...)`` and ``Model.objects.<method>(...)``."""
        # ``get_user_model()`` / ``django.contrib.auth.get_user_model()`` —
        # Django's runtime AUTH_USER_MODEL resolver. We can't read the
        # project's setting statically, but the builtin contrib User is
        # the right default; a workspace ``User`` model shadows it via
        # ``DjangoIndex.lookup`` anyway.
        if _is_get_user_model_call(value):
            return self.django_index.lookup("User")
        # Model(...) — direct instantiation.
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            return self.django_index.lookup(value.func.id)
        # Model.objects.<method>(...) — manager call.
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            method = value.func.attr
            if method not in _QUERY_METHODS_RETURNING_INSTANCE:
                return None
            mgr = value.func.value
            if (
                isinstance(mgr, ast.Attribute)
                and mgr.attr in {"objects", "_default_manager", "_base_manager"}
                and isinstance(mgr.value, ast.Name)
            ):
                return self.django_index.lookup(mgr.value.id)
        return None

    def _attr_is_magic(self, model: ModelInfo, attr_name: str) -> bool:
        cfg = self.config

        for group in ("manager", "meta", "exception"):
            if cfg.is_rule_enabled(group) and attr_name in cfg.merged_static_attrs(group):
                return True

        if cfg.is_rule_enabled("pk"):
            if attr_name in cfg.merged_static_attrs("pk"):
                # Special-case `id`: only present implicitly when no
                # explicit PK. With an explicit PK, the actual PK name
                # is whatever the user declared (see below).
                if attr_name == "id" and not model.implicit_id:
                    return False
                return True
            # The model's actual PK field name — `id` for implicit PK,
            # otherwise the field declared with ``primary_key=True``.
            # Suppressing this means ty's "unresolved-attribute" warning
            # for the real PK field never reaches the user.
            if attr_name == model.pk_name:
                return True

        if cfg.is_rule_enabled("fk_id") and attr_name in model.fk_id_accessors:
            return True

        if cfg.is_rule_enabled("reverse") and attr_name in self.django_index.reverse_attrs(model.qualname):
            return True

        if cfg.is_rule_enabled("generated") and attr_name in model.generated_method_names:
            return True

        return False

    def is_false_positive(self, uri: str, diagnostic: Diagnostic) -> bool:  # type: ignore[override]
        if not self.config.enabled:
            return False
        if (
            _is_unused_request(diagnostic)
            and self.config.is_rule_enabled("unused_request_param")
        ):
            try:
                return self._is_first_request_param(uri, diagnostic)
            except Exception:
                _log.exception("unused-request check crashed; keeping the diagnostic")
                return False
        if (
            _is_choices_enum_member_assignment(diagnostic)
            and self.config.is_rule_enabled("choices_enum")
        ):
            try:
                return self._is_in_choices_class(uri, diagnostic)
            except Exception:
                _log.exception("choices-enum check crashed; keeping the diagnostic")
                return False
        if (
            _is_unsupported_operator(diagnostic)
            and self.config.is_rule_enabled("f_operator")
        ):
            try:
                return self._is_combinable_operator(uri, diagnostic)
            except Exception:
                _log.exception("f-operator check crashed; keeping the diagnostic")
                return False
        if not _is_unresolved_attribute(diagnostic):
            return False
        try:
            return self._evaluate(uri, diagnostic)
        except Exception:
            _log.exception("analyzer crashed; keeping the diagnostic")
            return False

    def _is_in_choices_class(self, uri: str, diagnostic: Diagnostic) -> bool:
        path = _uri_to_path(uri)
        if path is None:
            return False
        parsed = self._parse(uri, path)
        if parsed is None:
            return False
        rng = diagnostic.get("range") or {}
        start = rng.get("start") or {}
        line_no = int(start.get("line", 0)) + 1
        col = int(start.get("character", 0))
        # Synthesize a fake target node and reuse the enclosing-class
        # walker — it only reads lineno.
        target = ast.AST()
        target.lineno = line_no
        target.col_offset = col
        cls = _enclosing_class(parsed.tree, target)
        if cls is None:
            return False
        return any(_base_is_choices(b) for b in cls.bases)

    def _is_combinable_operator(self, uri: str, diagnostic: Diagnostic) -> bool:
        """Suppress ty's ``unsupported-operator`` when either operand is a
        Django :class:`Combinable` expression.

        ``F('x') + timedelta(...)``, ``datetime.now() - F('x')``,
        ``-F('x')``, ``F('a') * F('b')``, ``Value(1) + F('x')``… Django's
        ``Combinable`` base overloads the arithmetic operators (plus
        ``__neg__`` and ``__invert__``) and swallows whatever's on the
        other side into ``Value(...)``. ty doesn't know that — when it
        sees ``datetime - F``, it checks ``datetime.__sub__``, finds no
        ``F`` overload, and emits a false positive. Python would fall
        back to ``F.__rsub__`` at runtime, which accepts anything.
        """
        path = _uri_to_path(uri)
        if path is None:
            return False
        parsed = self._parse(uri, path)
        if parsed is None:
            return False
        op_node = _find_op_at(parsed.tree, diagnostic.get("range") or {})
        if op_node is None:
            return False
        return _op_involves_combinable(op_node)

    def _is_first_request_param(self, uri: str, diagnostic: Diagnostic) -> bool:
        path = _uri_to_path(uri)
        if path is None:
            return False
        parsed = self._parse(uri, path)
        if parsed is None:
            return False
        rng = diagnostic.get("range") or {}
        start = rng.get("start") or {}
        # LSP positions are 0-indexed; AST line numbers are 1-indexed.
        line_no = int(start.get("line", 0)) + 1
        col = int(start.get("character", 0))
        for node in ast.walk(parsed.tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = list(node.args.posonlyargs) + list(node.args.args)
            # Skip self/cls so class-based views (`def get(self, request, ...)`)
            # are still treated as having `request` first.
            if params and params[0].arg in ("self", "cls"):
                params = params[1:]
            if not params:
                continue
            first = params[0]
            if first.arg != "request":
                continue
            if first.lineno == line_no and first.col_offset == col:
                return True
        return False

    def additional_diagnostics(self, uri: str) -> list[Diagnostic]:
        if not self.config.is_rule_enabled("orm_lookup"):
            return []
        if not self.django_index.models:
            return []
        path = _uri_to_path(uri)
        if path is None:
            return []
        parsed = self._parse(uri, path)
        if parsed is None:
            return []
        try:
            return list(self._scan_lookups(parsed))
        except Exception:
            _log.exception("orm-lookup scanner crashed; emitting nothing")
            return []

    def completions(self, uri: str, position: dict) -> CompletionResult:
        """Return LSP ``CompletionItem`` dicts for Django-aware positions.

        Two kinds of completion:

        * **ORM-lookup kwargs** — cursor inside
          ``Model.objects.<method>(...)`` at a kwarg slot. Result is
          *exclusive*: ty's name-of-any-variable completions are noise
          there.
        * **FK ``_id`` accessors** on attribute access — cursor after
          ``<receiver>.`` where the receiver resolves to a Django model.
          Result is *non-exclusive*: we augment ty's items with the
          ``<field>_id`` accessors it doesn't know about.

        On any uncertainty we return an empty non-exclusive result and
        let ty's completions through.
        """
        empty = CompletionResult()
        if not self.django_index.models:
            return empty
        path = _uri_to_path(uri)
        if path is None:
            return empty
        source = self._source_for(uri, path)
        if source is None:
            return empty
        try:
            if self.config.is_rule_enabled("orm_lookup"):
                result = self._scan_completions(source, position)
                if result.items or result.exclusive:
                    return result
            if self.config.is_rule_enabled("fk_id"):
                return self._scan_fk_id_completions(source, position)
            return empty
        except Exception:
            _log.exception("completion scanner crashed; emitting nothing")
            return empty

    def _scan_completions(self, source: str, position: dict) -> CompletionResult:
        empty = CompletionResult()
        line = int(position.get("line", 0))
        character = int(position.get("character", 0))
        offset = _offset_from_lsp_position(source, line, character)
        if offset > len(source):
            return empty

        # Walk back over identifier chars at the cursor to find the partial
        # kwarg name the user has typed so far (may be empty).
        partial_start = offset
        while partial_start > 0 and (
            source[partial_start - 1].isalnum()
            or source[partial_start - 1] == "_"
        ):
            partial_start -= 1
        partial = source[partial_start:offset]

        # Cheap precondition: the only places an ORM kwarg name can sit
        # are immediately after ``(`` or ``,`` (with whitespace allowed
        # in between for multi-line calls). Anything else — top-level
        # identifier, attribute access, the right-hand side of ``=``,
        # inside a dict literal — is definitely not ours. Bailing here
        # skips ~12 ms of buffer ast.parse on a 1k-line file. The
        # heuristic has a false negative for kwargs preceded by a
        # comment, which is rare enough to accept.
        if not _is_call_arg_position(source, partial_start):
            return empty

        # Patch the source so it parses: replace everything from the
        # partial onward with `<marker>=None`, then append closes to
        # balance any open brackets above the cursor. This turns an
        # in-progress `filter(em` into `filter(__marker__=None)` which
        # ast can chew on; the marker keyword is what we find later.
        marker = "__iommi_lsp_completion_marker__"
        head = source[:partial_start]
        inserted = marker + "=None"
        closes = _close_brackets(head + inserted)
        patched = head + inserted + closes

        try:
            tree = ast.parse(patched)
        except SyntaxError:
            return empty

        marker_call = _find_marker_call(tree, marker)
        if marker_call is None:
            return empty

        model: "ModelInfo | None" = None
        if (
            isinstance(marker_call.func, ast.Name)
            and marker_call.func.id in _HELPER_LOOKUP_FUNCS
            and marker_call.args
        ):
            model = self._helper_first_arg_model(marker_call.args[0], tree)
        elif isinstance(marker_call.func, ast.Attribute):
            method = marker_call.func.attr
            if method not in _LOOKUP_METHODS:
                return empty
            model = self._root_manager_model(marker_call.func.value, tree)
        else:
            return empty

        if model is None:
            # We recognised the call shape but can't say which model —
            # don't suppress ty here, the user might know better.
            return empty

        # `foreign_key__name` — walk the chain so completions reflect the
        # target model, not the receiver. The last `__`-segment is the
        # in-progress identifier; everything before it is the chain.
        if "__" in partial:
            head, _, suffix = partial.rpartition("__")
            chain = lookup_walker.split_chain(head)
            target = _walk_chain_for_completion(
                self.django_index, model, chain
            )
            if target is None:
                # Chain didn't resolve to something we can complete on
                # (unknown segment, or terminated at a non-relation leaf).
                # Still our position — bias toward exclusive empty.
                return CompletionResult(items=[], exclusive=True)
            items = list(_field_completion_items(
                self.django_index, target, suffix, prefix=head + "__"
            ))
        else:
            items = list(_field_completion_items(
                self.django_index, model, partial
            ))
        return CompletionResult(items=items, exclusive=True)

    def _scan_fk_id_completions(
        self, source: str, position: dict
    ) -> CompletionResult:
        """Suggest ``<field>_id`` accessors after ``<receiver>.``.

        Non-exclusive: ty handles the rest of the attribute completion
        (real fields, methods, etc.); we only contribute the FK
        underlying-column accessors that ty doesn't know about.
        """
        empty = CompletionResult()
        line = int(position.get("line", 0))
        character = int(position.get("character", 0))
        offset = _offset_from_lsp_position(source, line, character)
        if offset > len(source):
            return empty

        partial_start = offset
        while partial_start > 0 and (
            source[partial_start - 1].isalnum()
            or source[partial_start - 1] == "_"
        ):
            partial_start -= 1
        if partial_start == 0 or source[partial_start - 1] != ".":
            return empty
        partial = source[partial_start:offset]

        # Walk forward over identifier chars too, so a cursor in the
        # middle of `user_id` (`p.us|er_id`) replaces the whole token.
        forward_end = offset
        while forward_end < len(source) and (
            source[forward_end].isalnum()
            or source[forward_end] == "_"
        ):
            forward_end += 1

        marker = "__iommi_lsp_fk_id_marker__"
        patched = source[:partial_start] + marker + source[forward_end:]

        try:
            tree = ast.parse(patched)
        except SyntaxError:
            return empty

        target: ast.Attribute | None = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == marker:
                target = node
                break
        if target is None:
            return empty

        model = self._resolve_receiver_model(target.value, tree)
        if model is None:
            model = self._resolve_via_annotation(target.value, tree)
        if model is None:
            return empty

        items: list[dict] = []
        for name in sorted(model.fk_id_accessors):
            if partial and not name.startswith(partial):
                continue
            items.append({
                "label": name,
                "kind": 5,  # CompletionItemKind.Field
                "insertText": name,
                "detail": "FK underlying-column accessor",
                "data": {"source": "iommi_lsp.fk-id", "model": model.qualname},
            })
        # The model's actual PK name. For implicit-PK models that's ``id``
        # — Django adds it via the metaclass and ty has no way to see it.
        # For explicit-PK models the field is in the class body and ty
        # already knows about it, but offering it here is harmless: ty's
        # own completion items get merged on top.
        pk_name = model.pk_name
        if not partial or pk_name.startswith(partial):
            items.append({
                "label": pk_name,
                "kind": 5,
                "insertText": pk_name,
                "detail": "primary key",
                "data": {"source": "iommi_lsp.pk", "model": model.qualname},
            })
        return CompletionResult(items=items, exclusive=False)

    def _scan_lookups(self, parsed: _ParsedFile):
        for node in ast.walk(parsed.tree):
            if not isinstance(node, ast.Call):
                continue
            # ``get_object_or_404(Model, name__icontains=…)`` and friends —
            # plain function call where the first positional is the model.
            if isinstance(node.func, ast.Name) and node.func.id in _HELPER_LOOKUP_FUNCS:
                if not node.args:
                    continue
                model = self._helper_first_arg_model(node.args[0], parsed.tree)
                if model is None:
                    continue
                yield from self._validate_kwargs(parsed, model, node.keywords)
                for arg in node.args[1:]:
                    for q_kwargs in _iter_q_kwargs(arg):
                        yield from self._validate_kwargs(parsed, model, q_kwargs)
                yield from self._validate_f_calls(model, node)
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            method = node.func.attr
            if (
                method not in _LOOKUP_METHODS
                and method not in _FIELD_PATH_METHODS
                and method not in _AGGREGATE_METHODS
            ):
                continue
            model = self._root_manager_model(node.func.value, parsed.tree)
            if model is None:
                continue
            # Upstream ``annotate(alias=…)`` / ``aggregate(alias=…)`` /
            # ``alias(alias=…)`` declarations in the same expression
            # chain — those alias names are valid leaf lookups for the
            # filter/order_by/etc. that consumes them.
            aliases = _collect_chain_aliases(node)
            if method in _LOOKUP_METHODS:
                # `.filter(name__icontains=…)` — direct kwargs.
                yield from self._validate_kwargs(
                    parsed, model, node.keywords, aliases=aliases,
                )
                # `.filter(Q(a=1) | Q(b=2), …)` — kwargs inside Q expressions.
                for arg in node.args:
                    for q_kwargs in _iter_q_kwargs(arg):
                        yield from self._validate_kwargs(
                            parsed, model, q_kwargs, aliases=aliases,
                        )
            if method in _FIELD_PATH_METHODS:
                yield from self._validate_field_path_args(
                    parsed, model, node.args, method, aliases=aliases,
                )
            # F('field__path') anywhere in the call's args/kwargs.
            yield from self._validate_f_calls(model, node, aliases=aliases)

    def _helper_first_arg_model(
        self, arg: ast.AST, tree: ast.Module,
    ) -> ModelInfo | None:
        """Resolve the first positional arg of a helper-lookup call to a model.

        Accepts:
        * a bare model name (``get_object_or_404(User, …)``);
        * a module-qualified model (``get_object_or_404(myapp.models.User, …)``);
        * a manager-rooted queryset (``get_object_or_404(User.objects, …)``)
          or any chain of queryset methods on the same.
        """
        if isinstance(arg, ast.Name):
            return self.django_index.lookup(arg.id)
        if isinstance(arg, ast.Attribute):
            # ``foo.bar.Model`` — match the rightmost segment.
            return self.django_index.lookup(arg.attr)
        # Manager chain: ``User.objects.filter(...)`` → same resolver as
        # method-call form.
        return self._root_manager_model(arg, tree)

    def _validate_field_path_args(
        self,
        parsed: _ParsedFile,
        model: ModelInfo,
        args: list[ast.expr],
        method: str,
        aliases: frozenset[str] = frozenset(),
    ):
        for arg in args:
            # ``prefetch_related(Prefetch('rel', queryset=...))`` — the
            # first positional of the Prefetch call is a field path on
            # the receiver model.
            if (
                method == "prefetch_related"
                and isinstance(arg, ast.Call)
                and _is_prefetch_call(arg)
                and arg.args
            ):
                first = arg.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    chain = lookup_walker.split_chain(first.value)
                    result = lookup_walker.walk(
                        self.django_index, model.qualname, chain
                    )
                    if isinstance(result, lookup_walker.Problem):
                        diag = _string_problem_to_diagnostic(first, chain, 0, result)
                        if diag is not None:
                            yield diag
                continue
            if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                # F() / Prefetch() / variables — skip.
                continue
            raw = arg.value
            # `order_by('?')` — random ordering, not a field path.
            if method == "order_by" and raw == "?":
                continue
            leading = 0
            if method == "order_by" and raw.startswith("-"):
                leading = 1
            chain_str = raw[leading:]
            if not chain_str:
                continue
            chain = lookup_walker.split_chain(chain_str)
            if _chain_starts_with_alias(chain, aliases):
                continue
            result = lookup_walker.walk(self.django_index, model.qualname, chain)
            if isinstance(result, lookup_walker.Problem):
                diag = _string_problem_to_diagnostic(arg, chain, leading, result)
                if diag is not None:
                    yield diag

    def _validate_f_calls(
        self, model: ModelInfo, call: ast.Call,
        aliases: frozenset[str] = frozenset(),
    ):
        """Find F('field__path') anywhere in *call*'s arg/kwarg subtrees."""
        seen: set[int] = set()
        for sub in _iter_arg_subtrees(call):
            for fnode in ast.walk(sub):
                if not isinstance(fnode, ast.Call) or not _is_f_call(fnode):
                    continue
                key = id(fnode)
                if key in seen:
                    continue
                seen.add(key)
                if not fnode.args:
                    continue
                arg0 = fnode.args[0]
                if not isinstance(arg0, ast.Constant) or not isinstance(arg0.value, str):
                    continue
                chain = lookup_walker.split_chain(arg0.value)
                if _chain_starts_with_alias(chain, aliases):
                    continue
                result = lookup_walker.walk(self.django_index, model.qualname, chain)
                if isinstance(result, lookup_walker.Problem):
                    diag = _string_problem_to_diagnostic(arg0, chain, 0, result)
                    if diag is not None:
                        yield diag

    def _validate_kwargs(
        self,
        parsed: _ParsedFile,
        model: ModelInfo,
        kwargs: list[ast.keyword],
        aliases: frozenset[str] = frozenset(),
    ):
        for kw in kwargs:
            if kw.arg is None:
                continue   # **kwargs splat
            if kw.arg in _METHOD_ONLY_KWARGS:
                continue
            chain = lookup_walker.split_chain(kw.arg)
            if _chain_starts_with_alias(chain, aliases):
                continue
            result = lookup_walker.walk(self.django_index, model.qualname, chain)
            if isinstance(result, lookup_walker.Problem):
                diag = _problem_to_diagnostic(parsed.source, kw, chain, result)
                if diag is not None:
                    yield diag

    def _root_manager_model(
        self,
        receiver: ast.AST,
        tree: ast.Module | None = None,
        visited: frozenset[str] = frozenset(),
    ) -> ModelInfo | None:
        """Walk back through chained calls until we hit a manager-rooted form.

        Recognises three rooted shapes:

        * ``Model.<manager>`` — the canonical case (``User.objects``).
        * ``<dotted>.Model.<manager>`` — module-qualified access
          (``myapp.models.User.objects``); the rightmost segment is used
          as a simple-name lookup, which the index naturally
          ambiguity-protects.
        * A bare local variable previously assigned from one of the
          above, possibly through queryset-returning chains
          (``qs = User.objects.all(); qs.filter(...)``).

        Returns the model if the leftmost receiver is recognised; ``None``
        otherwise (which means we don't validate that call). The
        ``visited`` set guards against self-referential assignment loops.
        """
        cur = receiver
        # Peel off any chain of method calls: each call's func is an
        # Attribute whose value is the previous receiver.
        while isinstance(cur, ast.Call):
            if not isinstance(cur.func, ast.Attribute):
                return None
            cur = cur.func.value

        # `<…>.<manager>` — direct or module-qualified.
        if isinstance(cur, ast.Attribute):
            if cur.attr not in _MANAGER_NAMES:
                return None
            owner = cur.value
            if isinstance(owner, ast.Name):
                info = self.django_index.lookup(owner.id)
                if info is not None:
                    return info
                # Fall back to local-flow resolution: ``UserCls =
                # get_user_model()`` (or any other model-returning call)
                # rebinds the name, but ``UserCls.objects.filter(...)``
                # should still validate against the bound model.
                if tree is not None and owner.id not in visited:
                    return self._resolve_local_variable(owner.id, owner, tree)
                return None
            if isinstance(owner, ast.Attribute):
                # `models.User.objects` / `app.models.User.objects`.
                return self.django_index.lookup(owner.attr)
            return None

        # Bare name — could be a local queryset variable.
        if isinstance(cur, ast.Name) and tree is not None:
            if cur.id in visited:
                return None
            return self._resolve_local_queryset_model(
                cur.id, cur, tree, visited | {cur.id}
            )

        return None

    def _resolve_local_queryset_model(
        self,
        var_name: str,
        use_site: ast.AST,
        tree: ast.Module,
        visited: frozenset[str],
    ) -> ModelInfo | None:
        """Resolve a local variable to the model its queryset is bound to.

        Walks same-function assignments preceding *use_site*; the most
        recent successfully-resolved RHS wins. Recursion is bounded by
        *visited* (see :meth:`_root_manager_model`).
        """
        scope = _enclosing_function(tree, use_site)
        if scope is None:
            scope = tree
        last_match: ModelInfo | None = None
        use_pos = (
            getattr(use_site, "lineno", 0),
            getattr(use_site, "col_offset", 0),
        )
        for stmt in ast.walk(scope):
            if not isinstance(stmt, ast.Assign):
                continue
            if (stmt.lineno, stmt.col_offset) >= use_pos:
                continue
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name) and tgt.id == var_name:
                    inferred = self._root_manager_model(
                        stmt.value, tree, visited
                    )
                    if inferred is not None:
                        last_match = inferred
        return last_match


# ---------------------------------------------------------------------------
# Helpers — kept module-level so they're easy to test in isolation later.
# ---------------------------------------------------------------------------


def _node_contains(parent: ast.AST, target: ast.AST) -> bool:
    """Whether *target* is *parent* or a descendant of it (identity check)."""
    for sub in ast.walk(parent):
        if sub is target:
            return True
    return False


def _is_get_user_model_call(value: ast.AST) -> bool:
    """``get_user_model()`` / ``auth.get_user_model()`` / etc."""
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    if isinstance(func, ast.Name):
        return func.id == "get_user_model"
    if isinstance(func, ast.Attribute):
        return func.attr == "get_user_model"
    return False


def _collect_chain_aliases(call: ast.Call) -> frozenset[str]:
    """Walk back through ``call``'s receiver chain collecting alias names
    declared by sibling ``.annotate(...)`` / ``.aggregate(...)`` /
    ``.alias(...)`` calls.

    Scope is intentionally the *same expression chain*: aliases defined
    on a queryset variable upstream (``qs = User.objects.annotate(x=…);
    qs.filter(x=1)``) aren't picked up. The expression-chain case is the
    cheap, no-flow-analysis fix from FUTURE_PLANS.
    """
    aliases: set[str] = set()
    cur: ast.AST | None = (
        call.func.value if isinstance(call.func, ast.Attribute) else None
    )
    while isinstance(cur, ast.Call):
        func = cur.func
        if not isinstance(func, ast.Attribute):
            break
        if func.attr in _AGGREGATE_METHODS:
            for kw in cur.keywords:
                if kw.arg:
                    aliases.add(kw.arg)
        cur = func.value
    return frozenset(aliases)


def _chain_starts_with_alias(
    chain: list[str], aliases: frozenset[str],
) -> bool:
    """Treat ``alias`` and ``alias__<known_lookup>`` as valid leaves.

    Annotated values are scalar in Django — you can chain a transform/
    lookup (``myalias__gte``) but not a field traversal. The walker
    can't see local aliases, so we short-circuit here.
    """
    if not aliases or not chain:
        return False
    if chain[0] not in aliases:
        return False
    # Aliases that resolve to a relation (``Count('articles')`` returns
    # an int, but ``Subquery(qs)`` could return a row) are rare enough
    # in practice that we accept anything past the alias rather than
    # try to introspect the annotation's RHS.
    return True


def _is_unresolved_attribute(diagnostic: Diagnostic) -> bool:
    code = diagnostic.get("code")
    if isinstance(code, str) and code == "unresolved-attribute":
        return True
    # Some clients normalize ``code`` as an int; ty uses strings, but stay safe.
    if isinstance(code, dict) and code.get("value") == "unresolved-attribute":
        return True
    return False


_CHOICES_BASE_NAMES = frozenset({"Choices", "IntegerChoices", "TextChoices"})


def _is_choices_enum_member_assignment(diagnostic: Diagnostic) -> bool:
    """Match ty's ``invalid-assignment`` on an Enum member.

    ty has emitted at least two variants of this message across versions
    (``is incompatible with __new__`` and ``value is not assignable to
    expected type``). Both start with ``Enum member`` and share the
    diagnostic code, so we anchor on those.
    """
    code = diagnostic.get("code")
    code_value = code if isinstance(code, str) else (
        code.get("value") if isinstance(code, dict) else None
    )
    if code_value != "invalid-assignment":
        return False
    message = diagnostic.get("message")
    if not isinstance(message, str):
        return False
    return message.lstrip().startswith("Enum member")


def _base_is_choices(base: ast.expr) -> bool:
    if isinstance(base, ast.Name):
        return base.id in _CHOICES_BASE_NAMES
    if isinstance(base, ast.Attribute):
        return base.attr in _CHOICES_BASE_NAMES
    return False


# Django expression factories whose return value is a Combinable —
# i.e. arithmetic operators on the result combine into a CombinedExpression
# rather than going through the operand's regular dunder protocol.
# Slightly redundant with ``_STRING_FIELD_PATH_FUNCS`` (which is the
# subset that takes a ``"field__path"`` string) but listing them out
# here keeps the operator-filter intent obvious. ``ExpressionWrapper``
# and ``Cast`` wrap an inner expression and preserve Combinable-ness.
_COMBINABLE_FACTORY_NAMES: frozenset[str] = frozenset({
    "F", "Value", "Func", "OuterRef", "Subquery", "ExpressionWrapper",
    "Case", "When", "Cast",
    "Count", "Sum", "Avg", "Min", "Max", "StdDev", "Variance",
    "Coalesce", "Greatest", "Least", "Concat", "Length", "Lower", "Upper",
    "Substr", "Now", "Trunc", "TruncDate", "TruncTime",
    "TruncDay", "TruncMonth", "TruncYear",
    "TruncHour", "TruncMinute", "TruncSecond",
    "ExtractYear", "ExtractMonth", "ExtractDay",
    "ExtractWeek", "ExtractWeekDay", "ExtractIsoYear", "ExtractIsoWeekDay",
    "ExtractHour", "ExtractMinute", "ExtractSecond", "ExtractQuarter",
})


def _is_unsupported_operator(diagnostic: Diagnostic) -> bool:
    code = diagnostic.get("code")
    if isinstance(code, str) and code == "unsupported-operator":
        return True
    if isinstance(code, dict) and code.get("value") == "unsupported-operator":
        return True
    return False


def _expr_returns_combinable(node: ast.AST) -> bool:
    """Return True if *node* is, or evaluates to, a Django Combinable.

    Recurses through arithmetic ``BinOp`` / ``UnaryOp`` chains because
    Combinable.__add__ / __sub__ / etc. return another CombinedExpression
    — once F is involved, the whole chain is Combinable. Also recognises
    ``some_expr.bitand(other)``, ``.asc()``, ``.desc()`` and similar
    methods Django defines on Combinable.
    """
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id in _COMBINABLE_FACTORY_NAMES:
            return True
        if isinstance(func, ast.Attribute):
            if func.attr in _COMBINABLE_FACTORY_NAMES:
                return True
            # ``<expr>.bitand(...)``, ``<expr>.bitor(...)``, ``<expr>.asc()``…
            # — Combinable methods that keep the result Combinable.
            if func.attr in _COMBINABLE_METHOD_NAMES:
                return _expr_returns_combinable(func.value)
    if isinstance(node, ast.BinOp):
        return _expr_returns_combinable(node.left) or _expr_returns_combinable(node.right)
    if isinstance(node, ast.UnaryOp):
        return _expr_returns_combinable(node.operand)
    if isinstance(node, ast.Subscript):
        # ``F('name')[1:5]`` — string-slice on Combinable returns Combinable.
        return _expr_returns_combinable(node.value)
    return False


_COMBINABLE_METHOD_NAMES: frozenset[str] = frozenset({
    "bitand", "bitor", "bitxor", "bitleftshift", "bitrightshift",
    "asc", "desc",
})


def _op_involves_combinable(node: ast.AST) -> bool:
    """``BinOp`` / ``UnaryOp`` / ``Compare`` where any operand is Combinable."""
    if isinstance(node, ast.BinOp):
        return _expr_returns_combinable(node.left) or _expr_returns_combinable(node.right)
    if isinstance(node, ast.UnaryOp):
        return _expr_returns_combinable(node.operand)
    if isinstance(node, ast.Compare):
        if _expr_returns_combinable(node.left):
            return True
        return any(_expr_returns_combinable(c) for c in node.comparators)
    return False


def _find_op_at(tree: ast.Module, range_: dict) -> ast.AST | None:
    """Find the smallest ``BinOp``/``UnaryOp``/``Compare`` containing the LSP range."""
    start = range_.get("start") or {}
    end = range_.get("end") or {}
    s_line = int(start.get("line", 0)) + 1
    s_col = int(start.get("character", 0))
    e_line = int(end.get("line", s_line - 1)) + 1
    e_col = int(end.get("character", s_col))

    best: ast.AST | None = None
    best_size = (10**9, 10**9)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare)):
            continue
        nl = node.lineno
        nc = node.col_offset
        nel = node.end_lineno or nl
        nec = node.end_col_offset or nc
        if (nl, nc) > (s_line, s_col):
            continue
        if (nel, nec) < (e_line, e_col):
            continue
        size = (nel - nl, nec - nc)
        if size < best_size:
            best = node
            best_size = size
    return best


def _is_unused_request(diagnostic: Diagnostic) -> bool:
    """Match ty's hint ``\\`request\\` is unused``. ty emits this with no
    diagnostic code, so we sniff source + message text directly.
    """
    if diagnostic.get("source") != "ty":
        return False
    message = diagnostic.get("message")
    if not isinstance(message, str):
        return False
    return message.strip() == "`request` is unused"


def _uri_to_path(uri: str) -> Path | None:
    if not uri.startswith("file://"):
        return None
    parsed = urlparse(uri)
    return Path(unquote(parsed.path))


def _find_attribute_at(tree: ast.Module, range_: dict) -> ast.Attribute | None:
    """Find the smallest ``ast.Attribute`` node containing the LSP range."""
    start = range_.get("start") or {}
    end = range_.get("end") or {}
    s_line = int(start.get("line", 0)) + 1   # LSP is 0-indexed, AST is 1-indexed
    s_col = int(start.get("character", 0))
    e_line = int(end.get("line", s_line - 1)) + 1
    e_col = int(end.get("character", s_col))

    best: ast.Attribute | None = None
    best_size = (10**9, 10**9)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        nl = node.lineno
        nc = node.col_offset
        nel = node.end_lineno or nl
        nec = node.end_col_offset or nc
        # Node range must contain the diagnostic range.
        if (nl, nc) > (s_line, s_col):
            continue
        if (nel, nec) < (e_line, e_col):
            continue
        size = (nel - nl, nec - nc)
        if size < best_size:
            best = node
            best_size = size
    return best


def _is_f_call(call: ast.Call) -> bool:
    """Recognise ``F(...)`` and ``models.F(...)`` calls."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id in _STRING_FIELD_PATH_FUNCS
    if isinstance(func, ast.Attribute):
        return func.attr in _STRING_FIELD_PATH_FUNCS
    return False


def _is_prefetch_call(call: ast.Call) -> bool:
    """Recognise ``Prefetch(...)`` and ``models.Prefetch(...)`` calls."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "Prefetch"
    if isinstance(func, ast.Attribute):
        return func.attr == "Prefetch"
    return False


# Calls whose first positional argument is a Django field-path string —
# the same shape as ``F('author__name')``. Aggregates count/sum/avg/etc.
# accept additional ``filter=`` / ``distinct=`` kwargs after the path,
# but we only validate the path; everything else is left to ty.
_STRING_FIELD_PATH_FUNCS: frozenset[str] = frozenset({
    "F",
    "Count", "Sum", "Avg", "Min", "Max", "StdDev", "Variance",
    "OuterRef", "Subquery",
})


def _iter_arg_subtrees(call: ast.Call):
    """Yield the AST subtrees of *call*'s positional + keyword args.

    Avoids the ``func`` subtree so chained-receiver calls aren't
    re-scanned (each chained call is reached on its own ``ast.walk``).
    """
    for a in call.args:
        yield a
    for kw in call.keywords:
        if kw.value is not None:
            yield kw.value


def _is_q_call(call: ast.Call) -> bool:
    """Recognise ``Q(...)`` and ``models.Q(...)`` / ``...Q(...)`` calls."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "Q"
    if isinstance(func, ast.Attribute):
        return func.attr == "Q"
    return False


def _iter_q_kwargs(node: ast.AST):
    """Yield the keyword lists of every Q(...) call reachable from *node*.

    Walks through ``|`` / ``&`` (BinOp) and ``~`` (UnaryOp) since Q
    expressions compose via boolean operators. Bare ``Q`` references
    (variables, attribute access without a call) are ignored — we don't
    follow data flow.
    """
    if isinstance(node, ast.Call):
        if _is_q_call(node):
            yield node.keywords
            # Q(Q(a=1), b=2) — nested Q in positional args.
            for sub in node.args:
                yield from _iter_q_kwargs(sub)
        return
    if isinstance(node, ast.BoolOp):
        for v in node.values:
            yield from _iter_q_kwargs(v)
        return
    if isinstance(node, ast.BinOp):
        yield from _iter_q_kwargs(node.left)
        yield from _iter_q_kwargs(node.right)
        return
    if isinstance(node, ast.UnaryOp):
        yield from _iter_q_kwargs(node.operand)
        return


def _problem_to_diagnostic(
    source: str,
    kw: ast.keyword,
    chain: list[str],
    problem: lookup_walker.Problem,
) -> Diagnostic | None:
    """Pin the diagnostic to the bad segment within the kwarg name."""
    if kw.arg is None or kw.value is None:
        return None
    arg_name = kw.arg
    line0 = (kw.value.lineno - 1) if kw.value.lineno else 0
    lines = source.splitlines()
    line_text = lines[line0] if 0 <= line0 < len(lines) else ""
    # Anchor on `arg_name=` so a kwarg name that also appears earlier as
    # a value (rare, but possible) doesn't mis-pin us.
    needle = f"{arg_name}="
    name_col = line_text.find(needle)
    if name_col == -1:
        name_col = line_text.find(arg_name)

    if name_col == -1:
        col_start = kw.value.col_offset or 0
        col_end = col_start + 1
        return _make_orm_diagnostic(
            line0, col_start, col_end, _format_orm_message(problem), problem
        )

    sep = "__"
    seg_offset = 0
    for i, seg in enumerate(chain):
        if i == problem.segment_index:
            break
        seg_offset += len(seg) + len(sep)

    col_start = name_col + seg_offset
    col_end = col_start + len(problem.bad_segment)
    return _make_orm_diagnostic(
        line0, col_start, col_end, _format_orm_message(problem), problem
    )


def _string_problem_to_diagnostic(
    arg: ast.Constant,
    chain: list[str],
    leading: int,
    problem: lookup_walker.Problem,
) -> Diagnostic | None:
    """Pin a diagnostic to the bad segment inside a string-literal field path.

    *leading* is the count of source characters consumed before the chain
    begins (e.g. ``1`` for ``order_by('-foo')`` to skip the ``-``).
    """
    if arg.lineno is None or arg.col_offset is None:
        return None
    line0 = arg.lineno - 1
    # `arg.col_offset` points at the opening quote of the string literal.
    # Adding 1 skips the quote; works for normal `'...'` / `"..."`.
    # Triple-quoted or implicit-concat literals can produce slightly
    # off offsets — we accept that as a cosmetic edge case.
    quote_skip = 1
    seg_offset = 0
    for i, seg in enumerate(chain):
        if i == problem.segment_index:
            break
        seg_offset += len(seg) + len("__")
    col_start = arg.col_offset + quote_skip + leading + seg_offset
    col_end = col_start + len(problem.bad_segment)
    return _make_orm_diagnostic(
        line0, col_start, col_end, _format_orm_message(problem), problem
    )


def _format_orm_message(problem: lookup_walker.Problem) -> str:
    if problem.outcome == "unknown_field":
        msg = (
            f"unknown ORM field/relation {problem.bad_segment!r} on "
            f"{problem.on_model}"
        )
        if problem.available:
            hint = ", ".join(problem.available[:8])
            if len(problem.available) > 8:
                hint += ", …"
            msg += f"  (available: {hint})"
        return msg
    if problem.outcome == "unknown_lookup":
        return (
            f"unknown ORM lookup {problem.bad_segment!r} after a leaf field "
            f"on {problem.on_model}"
        )
    return f"invalid ORM lookup chain at {problem.bad_segment!r}"


def _make_orm_diagnostic(
    line: int,
    col_start: int,
    col_end: int,
    message: str,
    problem: lookup_walker.Problem,
) -> Diagnostic:
    return {
        "code": _ORM_LOOKUP_DIAG_CODE,
        "message": message,
        "range": {
            "start": {"line": line, "character": col_start},
            "end": {"line": line, "character": col_end},
        },
        "severity": 2,   # warning — bias toward false negatives
        "source": _ORM_LOOKUP_DIAG_SOURCE,
        "data": {
            "outcome": problem.outcome,
            "on_model": problem.on_model,
            "available": list(problem.available),
        },
    }


def _offset_from_lsp_position(text: str, line: int, character: int) -> int:
    """Convert LSP ``{line, character}`` to a Python ``str`` offset.

    LSP characters are UTF-16 code units; non-BMP code points (emoji)
    count as two. For ASCII Python source — the overwhelming common
    case — this collapses to straight character indexing.
    """
    offset = 0
    cur_line = 0
    n = len(text)
    while offset < n and cur_line < line:
        if text[offset] == "\n":
            cur_line += 1
        offset += 1
    char_units = 0
    while offset < n and char_units < character:
        ch = text[offset]
        if ch == "\n":
            break
        char_units += 2 if ord(ch) > 0xFFFF else 1
        offset += 1
    return offset


def _is_call_arg_position(source: str, partial_start: int) -> bool:
    """Cheap upper bound on whether *partial_start* sits at a fresh
    call-argument name slot.

    The only characters that can immediately precede a new positional or
    kwarg name are ``(`` (first arg) and ``,`` (subsequent args), with
    arbitrary whitespace in between (multi-line calls are common). If
    the previous non-whitespace character is anything else — a newline
    that lands on a non-call line, ``.``, ``=``, ``:``, identifier
    chars, etc. — the cursor is definitely not at a kwarg name slot and
    we can skip the buffer parse the heavy path would otherwise do.
    """
    i = partial_start - 1
    while i >= 0 and source[i].isspace():
        i -= 1
    if i < 0:
        return False
    return source[i] in "(,"


def _close_brackets(src: str) -> str:
    """Return the closing tokens needed to balance *src*.

    Best-effort string-aware scan — triple-quoted strings and f-strings
    can confuse the cursor over multiple lines, but the resulting parse
    just fails and completion is suppressed.
    """
    stack: list[str] = []
    pair = {"(": ")", "[": "]", "{": "}"}
    in_string: str | None = None
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        if in_string is not None:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == in_string:
                in_string = None
            i += 1
            continue
        if ch in '"\'':
            in_string = ch
        elif ch in "([{":
            stack.append(pair[ch])
        elif ch in ")]}":
            if stack and stack[-1] == ch:
                stack.pop()
        elif ch == "#":
            j = src.find("\n", i)
            i = n if j == -1 else j
            continue
        i += 1
    return "".join(reversed(stack))


def _find_marker_call(tree: ast.AST, marker: str) -> ast.Call | None:
    """Return the smallest ``Call`` that has *marker* as a keyword name."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == marker:
                return node
    return None


def _field_completion_items(index, model, partial: str, prefix: str = ""):
    """Yield ``CompletionItem``-shaped dicts for *model*'s queryable names.

    Combines declared fields, FK-id accessors, the ``pk`` alias, and
    reverse-relation accessors. Each item carries ``insertText=name=``
    so accepting a completion drops the cursor right after the ``=``.

    When *prefix* is non-empty (a ``foo__`` chain typed before the cursor),
    the label and insertText include it so the editor replaces the whole
    word — without that, accepting a completion would clobber the chain.
    """
    items: dict[str, dict] = {}
    for name in model.fields:
        items[name] = {
            "label": name,
            "detail": _field_detail(model.fields[name]),
        }
    for name in model.fk_id_accessors:
        items.setdefault(name, {
            "label": name,
            "detail": "FK underlying-column accessor",
        })
    items.setdefault("pk", {"label": "pk", "detail": "primary key alias"})
    for name, source in (index.reverse_relations.get(model.qualname) or {}).items():
        items.setdefault(name, {
            "label": name,
            "detail": f"reverse relation → {source}",
        })

    for name in sorted(items):
        if partial and not name.startswith(partial):
            continue
        item = items[name]
        full = f"{prefix}{name}"
        yield {
            "label": full,
            "kind": 5,  # CompletionItemKind.Field
            "insertText": f"{full}=",
            "detail": item["detail"],
            "data": {"source": "iommi_lsp.orm-kwarg", "model": model.qualname},
        }


def _walk_chain_for_completion(
    index, start: "ModelInfo", chain: list[str]
) -> "ModelInfo | None":
    """Walk *chain* from *start* and return the model to complete on.

    Returns the target model when every segment is a relation (forward
    FK/OneToOne/M2M or reverse accessor) and the chain ends on one. Any
    non-relation segment (concrete field, ``pk``, FK-id accessor, ORM
    lookup name) terminates the walk with ``None`` — there's nothing
    meaningful to complete past a scalar.

    Mirrors the conventions of :mod:`lookup_walker` but is structured for
    completion rather than validation: unknown segments also return
    ``None`` so we don't suppress ty with bogus suggestions.
    """
    current: "ModelInfo | None" = start
    for seg in chain:
        if current is None:
            return None
        if seg == "pk" or seg in current.fk_id_accessors:
            return None
        fi = current.fields.get(seg)
        if fi is None:
            source = index.reverse_source(current.qualname, seg)
            if source is None:
                return None
            current = index.models.get(source)
            continue
        if fi.field_type not in RELATION_FIELD_NAMES:
            return None
        target = fi.target
        if target is None:
            return None
        current = index.models.get(target)
    return current


def _field_detail(fi) -> str:
    if fi.target:
        return f"{fi.field_type} → {fi.target}"
    return fi.field_type


def _enclosing_function(tree: ast.Module, target: ast.AST) -> ast.AST | None:
    target_line = getattr(target, "lineno", None)
    if target_line is None:
        return None
    best: ast.AST | None = None
    best_span = 10**9
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.lineno is None or node.end_lineno is None:
            continue
        if not (node.lineno <= target_line <= node.end_lineno):
            continue
        span = node.end_lineno - node.lineno
        if span < best_span:
            best = node
            best_span = span
    return best


def _enclosing_class(tree: ast.Module, target: ast.AST) -> ast.ClassDef | None:
    target_line = getattr(target, "lineno", None)
    if target_line is None:
        return None
    best: ast.ClassDef | None = None
    best_span = 10**9
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.lineno is None or node.end_lineno is None:
            continue
        if not (node.lineno <= target_line <= node.end_lineno):
            continue
        span = node.end_lineno - node.lineno
        if span < best_span:
            best = node
            best_span = span
    return best
