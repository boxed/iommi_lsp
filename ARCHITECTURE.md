# iommi_lsp — Architecture

How the analyzers work internally. For the user-facing feature list and
configuration, see the [README](README.md). For the higher-level design
rationale (why a proxy, why subtractive diagnostics), see
[DESIGN.md](DESIGN.md).

## How the iommi analyzer works

`iommi_lsp graph build` (or the auto-build at startup) imports iommi
in your venv and walks each `Refinable`-declaring class (`Table`,
`Column`, `Form`, `Field`, …) transitively through `class_ref` and
`members` edges. Every refinable gets one of these kinds:

* `members` — open dict of typed values (`columns: Dict[str, Column]`)
* `html_attrs` — the `attrs` special with `class` (str→bool) and `style`
  (str→str) sub-namespaces
* `class_ref` — chain steps into another refinable class (annotation
  wins over runtime default, so `bulk: Optional[Form]` resolves to Form)
* `traditional_class` — steps into a non-refinable class whose
  configurable surface is its `__init__`'s `self.X = …` assignments
  (e.g. `Column.cell` / `Table.cell` configuring a `Cell` instance)
* `namespace` — structured with a small set of known sub-keys
* `open_namespace` — empty Namespace default; any keys allowed
* `evaluated_scalar` / `scalar` — leaf; chain ends here

### Diagnostics

At LSP time the analyzer finds every `Class(kw__chain=...)` call, splits
the kwarg name on `__`, and walks the chain through the graph. Dead
ends become `iommi-unknown-refinable` warnings pinned to the offending
segment. Bias is toward false negatives — if anything is ambiguous
(unknown root class, member with no typed value, custom user subclass
not in the graph), we pass silently.

### Completions

At a recognised iommi-call kwarg position the LSP claims **exclusivity**:
ty's free-form variable suggestions are dropped so you only see real
refinables. Three flavours of completion fire from the same position:

* **Refinable names** — `Table(c‸` suggests `columns__`, `cell__`,
  `query__`, … with container refinables getting a trailing `__` and
  scalars getting `=`.
* **`auto__` namespace** — synthesised as a known namespace with
  `model` / `rows` / `instance` / `include` / `exclude`, even when the
  reflected graph records `auto` as an open namespace.
* **Django field names** inside `auto__include=[...]` /
  `auto__exclude=[...]` string literals, and after `columns__` /
  `fields__` / `filters__` / `parts__` when the call carries
  `auto__model=Model` (or `auto__rows=Model.objects.…`). This is the
  bridge that turns `Table(auto__model=User, columns__‸)` into a
  member-name list drawn from the `User` model's fields.

Synthesised stubs cover `Table`, `Form`, `Query`, and `Page` so the
above all works before `iommi_lsp graph build` ever succeeds; the
project's own iommi subclasses light up once a real graph is available.

## How the Django filter works

For each `unresolved-attribute` diagnostic from `ty`:

1. Read the file at the diagnostic's range and find the `<receiver>.<attr>`
   AST node.
2. Resolve the receiver type by **syntactic match** (`User.objects`) or
   **same-function local flow** (`u = User.objects.get(...); u.pk`).
3. If the receiver is a known model from the workspace's AST-only Django
   index *and* the attribute is metaclass-injected (`objects`, `_meta`,
   `pk`/`id`, `<fk>_id`, a known reverse relation, etc.), **drop**. Otherwise
   forward unchanged.

Bias is explicitly toward false negatives — better to leak a bit of noise
than to suppress a real bug. The index is rebuilt incrementally on
`didChange`/`didSave` and never imports the user's code.

## Unknown ORM field/lookup diagnostics

On top of subtracting ty's false positives, the Django analyzer emits its
own `django-unknown-orm-lookup` warnings when it spots a kwarg or string
path that does not resolve against the workspace index. The intent is to
catch typos that the type checker can't see — `User.objects.filter(eemail='x')`
is a valid Python call, but a SQL error at runtime.

Covered call shapes:

* kwargs on `filter` / `exclude` / `get` / `get_or_create` /
  `update_or_create` / `update` / `create`, including `__`-traversal
  through relation fields and reverse relations;
* string field paths in `order_by` / `values` / `values_list` / `only` /
  `defer` / `distinct` / `select_related` / `prefetch_related`
  (`order_by('-foo')` strips the leading `-`; `'?'` is recognised);
* `Q(field=…)` / `models.Q(…)` reachable through `|` / `&` / `~`
  composition, including nested `Q(Q(…), …)`;
* `F('field__path')` anywhere in the call's args or kwargs.

Receivers we recognise (anything else is silent):

* `Model.objects.…` and friends (`_default_manager`, `_base_manager`);
* `pkg.Model.objects.…` / `myapp.models.Model.objects.…` — the rightmost
  attribute segment is matched against the index by simple name, so it
  inherits the index's natural ambiguity protection (two models with the
  same simple name → silent);
* local variables previously assigned from any of the above within the
  same function (or at module scope), including chained reassignments
  like `qs = User.objects.all(); qs = qs.filter(...); qs.filter(...)`.

Bias is the same as the subtractive filter — when the receiver is
unknown, ambiguous, or comes from a parameter / queryset method we don't
model, we say nothing rather than risk a false positive. Disable
entirely with:

```toml
[tool.iommi_lsp]
disabled_rules = ["orm_lookup"]
```

### ORM-kwarg completion

When the cursor is inside `Model.objects.<lookup_method>(...)` (the
same method set the diagnostic covers — `filter`/`exclude`/`get`/
`update`/`create`/`get_or_create`/`update_or_create`) and you've typed
the start of a kwarg name, the LSP suggests every queryable name on
the model: declared fields, FK `_id` accessors, the `pk` alias, and
reverse-relation accessors. Each suggestion inserts as `name=` so the
caret lands inside the value position.

Triggered by `(` and `,` (and continuous-completion mode in most
editors). The receiver-resolution rules match the diagnostic path —
direct `Model.objects.…`, module-qualified `pkg.Model.objects.…`, and
local queryset variables (`qs = User.objects.all(); qs.filter(em‸)`).

At a recognised position we claim **exclusivity**: ty's items are
dropped from the response so the user doesn't see noise like `em`
matching any random local variable next to our `email=`. Empty +
exclusive is intentional — if the partial matches no field, the
editor shows nothing rather than back-filling with ty's free-form
name list. When the receiver doesn't resolve (`qs.filter(em‸` where
we can't tell what `qs` is) we step back and let ty handle it. If
ty errored on the completion request we substitute our own response;
if ty responded normally we either replace (exclusive) or merge
(non-exclusive).

### Built-in models and inheritance

The index bundles a static stub of the Django contrib models so projects
that import `django.contrib.auth.models.User`, `Group`, `Permission`,
`django.contrib.contenttypes.models.ContentType`, or
`django.contrib.sessions.models.Session` get validation without us
having to import site-packages. A workspace model with the same simple
name (e.g. a custom `User` via `AUTH_USER_MODEL`) shadows the builtin
during name resolution.

Abstract-base fields propagate to concrete subclasses. So
`class User(AbstractUser): ...` correctly resolves `email` /
`username` / etc., and your own `class Timestamped(models.Model):
class Meta: abstract = True` lets a `Book(Timestamped)` filter on
`created` without a false positive.
