# iommi-lsp

A wrapper Language Server that proxies [`ty`](https://github.com/astral-sh/ty)
and adds the Django- and iommi-awareness ty can't have on its own.

## What you get

### Django

* **Real autocomplete for ORM kwargs.** Inside `User.objects.filter(|`,
  `.exclude(|`, `.get(|`, `.update(|`, `.create(|`, `.get_or_create(|`,
  `.update_or_create(|` you get the model's queryable names — declared
  fields, FK `_id` accessors, `pk`, reverse-relation accessors —
  with `__`-traversal into related models. Suggestions insert as
  `name=` so the caret lands at the value. At a recognised call site
  we claim **exclusivity**, so ty's "any local variable near `em`"
  noise stays out of the list.
* **Typo diagnostics ty can't see.** `django-unknown-orm-lookup`
  warnings fire on kwargs and string field paths that don't resolve
  against the workspace model index. Covers `.filter(...)` /
  `.exclude(...)` / `.get(...)` and friends, `order_by` / `values` /
  `values_list` / `only` / `defer` / `distinct` / `select_related` /
  `prefetch_related` strings, `Q(...)` / `F('...')` expressions —
  full `__`-traversal through relations and reverse relations.
* **No more `Item.objects` false positives.** ty's
  `unresolved-attribute` diagnostics on Django metaclass magic
  (`objects`, `_meta`, `pk`/`id`, `<fk>_id`, reverse relations,
  `DoesNotExist`, …) are dropped before they reach the editor. Real
  bugs survive.
* **Built-in models + abstract inheritance.** `django.contrib.auth`
  / `contenttypes` / `sessions` models are stubbed so they work out
  of the box, and abstract-base fields propagate to concrete
  subclasses (so a custom `User(AbstractUser)` resolves `email` /
  `username` / etc.).

### iommi

* **Refinable autocomplete inside `Class(kw__chain=...)` calls.**
  `Table(c|` suggests `columns__`, `cell__`, `query__`, …;
  containers get a trailing `__` and scalars get `=`. Chains walk
  the iommi refinable graph, so `Table(columns__name__|` offers
  the configurable surface of `Column`.
* **`auto__` namespace.** Always surfaces `model` / `rows` /
  `instance` / `include` / `exclude` whether or not the graph
  reflects it, since iommi's default `Namespace()` is empty.
* **Django field bridging.** `Table(auto__model=User, columns__|)`
  suggests `User`'s fields (insert as `username__`, `email__`, …
  so you can keep configuring the auto-generated column). The same
  works inside `auto__include=['|']` / `auto__exclude=['|']` string
  literals.
* **`iommi-unknown-refinable` diagnostics.** Invalid chains in
  `Class(kw__chain=...)` calls flag the first dead-end segment.
* **Zero-setup defaults.** Synthesised stubs cover the public iommi
  classes (`Table`, `Form`, `Query`, `Page`) so all of the above
  works before any graph build succeeds; the project's own iommi
  subclasses light up once a real graph is built (automatically, in
  most setups — see below).

It speaks plain LSP, runs on stdio, and is configured into your editor
in place of `ty server`. See [DESIGN.md](DESIGN.md) for the
architecture.

## Status

Pre-1.0. Pinned against a narrow ty range — bumps are gated by a
contract test suite (`tests/test_contract_real_ty.py`).

## Install

```sh
uv tool install iommi-lsp     # or: pipx install iommi-lsp
```

`ty` must be available on `PATH` (`uv tool install ty` works), or pass
`--ty-command "uvx ty server"` to use a one-shot uv invocation.

## Run

```sh
iommi-lsp                                    # spawns `ty server` from PATH
iommi-lsp --ty-command "uvx ty server"       # uvx fallback
iommi-lsp --workspace ./myproject            # eager indexing for debugging
iommi-lsp index ./myproject                  # dump the Django model index and exit
iommi-lsp graph build ./myproject            # reflect installed iommi -> .iommi-lsp-graph.json
```

**For the iommi analyzer**, the graph at `.iommi-lsp-graph.json` is built
automatically when the workspace is opened:

1. **In-process** if `iommi` is importable from `iommi-lsp`'s interpreter
   (i.e. installed alongside it: `uv tool install --with iommi iommi-lsp`).
2. **Subprocess** against the workspace's `.venv` / `venv` Python, when
   `iommi-lsp` is installed there too.
3. **Synthesized stubs** for the well-known iommi classes (`Table`,
   `Form`, `Query`, `Page`) as a last resort — enough that `auto__…` and
   members-name completion still work before any graph build succeeds.

Running `iommi-lsp graph build` by hand is still supported and is the
fastest way to force a rebuild after upgrading iommi. The graph is a few
hundred KB JSON in your workspace root; check it in or `.gitignore` it
as you prefer.

`iommi-lsp` writes diagnostics-side stderr logs; tune via
`IOMMI_LSP_LOG=DEBUG` or `--log-level DEBUG`.

## Editor configuration

### Neovim (`nvim-lspconfig` style)

```lua
local lspconfig = require("lspconfig")
local configs = require("lspconfig.configs")

if not configs.iommi_lsp then
  configs.iommi_lsp = {
    default_config = {
      cmd = { "iommi-lsp" },
      filetypes = { "python" },
      root_dir = lspconfig.util.root_pattern("pyproject.toml", ".git"),
      single_file_support = false,
    },
  }
end

lspconfig.iommi_lsp.setup({})
```

### Helix (`languages.toml`)

```toml
[language-server.iommi-lsp]
command = "iommi-lsp"

[[language]]
name = "python"
language-servers = ["iommi-lsp"]
```

### Zed (`settings.json` under `lsp`)

```json
{
  "lsp": {
    "iommi-lsp": {
      "binary": { "path": "iommi-lsp" }
    }
  },
  "languages": {
    "Python": {
      "language_servers": ["iommi-lsp"]
    }
  }
}
```

### VS Code

There's no first-party VS Code extension yet. The simplest path is the
[`vscode-generic-lsp-client`](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
pattern: install a generic LSP-client extension and point it at
`iommi-lsp`. A first-party extension is on the roadmap.

## How the iommi analyzer works

`iommi-lsp graph build` (or the auto-build at startup) imports iommi
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

* **Refinable names** — `Table(c|` suggests `columns__`, `cell__`,
  `query__`, … with container refinables getting a trailing `__` and
  scalars getting `=`.
* **`auto__` namespace** — synthesised as a known namespace with
  `model` / `rows` / `instance` / `include` / `exclude`, even when the
  reflected graph records `auto` as an open namespace.
* **Django field names** inside `auto__include=[...]` /
  `auto__exclude=[...]` string literals, and after `columns__` /
  `fields__` / `filters__` / `parts__` when the call carries
  `auto__model=Model` (or `auto__rows=Model.objects.…`). This is the
  bridge that turns `Table(auto__model=User, columns__|)` into a
  member-name list drawn from the `User` model's fields.

Synthesised stubs cover `Table`, `Form`, `Query`, and `Page` so the
above all works before `iommi-lsp graph build` ever succeeds; the
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
[tool.iommi-lsp]
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
local queryset variables (`qs = User.objects.all(); qs.filter(em|)`).

At a recognised position we claim **exclusivity**: ty's items are
dropped from the response so the user doesn't see noise like `em`
matching any random local variable next to our `email=`. Empty +
exclusive is intentional — if the partial matches no field, the
editor shows nothing rather than back-filling with ty's free-form
name list. When the receiver doesn't resolve (`qs.filter(em|` where
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

## Per-project configuration

Add a `[tool.iommi-lsp]` table to your `pyproject.toml`:

```toml
[tool.iommi-lsp]
enabled = true                          # master switch
disabled_rules = ["pk", "reverse"]       # skip rule groups for this project

[tool.iommi-lsp.extra_magic_attrs]
manager = ["mongo", "search"]            # treat these as Manager-like attrs
```

Recognised rule groups: `manager`, `meta`, `pk`, `exception`, `fk_id`,
`reverse`, `orm_lookup`. Unknown groups in `disabled_rules` are ignored
with a stderr warning rather than silently breaking the filter.

A missing or malformed `pyproject.toml` falls back to defaults; the
proxy never crashes on a bad config.

## Caveats

* **Pre-1.0 ty.** Diagnostic codes and message text *will* change. The
  contract suite (`tests/test_contract_real_ty.py`) catches breakage when
  you bump ty.
* **iommi graph requires iommi to be importable somewhere.** Either in
  the same venv as `iommi-lsp`, or in the workspace's `.venv` / `venv`.
  Without that, the synthesised stubs cover the public iommi classes
  but project-specific subclasses (and their refinables) stay invisible
  until you run `iommi-lsp graph build`.
* **No type-checker arbitrage.** This proxies one backend at a time; you
  still pick `ty` (or eventually `mypy` / `pyright` once those backends
  are wired in).
* **Astral may absorb this.** If/when `ty` ships first-class library
  support, the Django filter becomes mostly redundant. The proxy and the
  iommi layer remain useful.

## Development

```sh
uv venv
uv pip install -e ".[dev]"
uv run pytest
```
