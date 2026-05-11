# iommi-lsp

A wrapper Language Server that proxies [`ty`](https://github.com/astral-sh/ty)
and filters out the ``unresolved-attribute`` false positives that Django's
metaclass magic produces — `Item.objects`, `Item._meta`, `item.pk`, reverse
relations, FK `_id` accessors, etc. The genuine bugs survive.

It speaks plain LSP, runs on stdio, and is configured into your editor in
place of `ty server`. See [DESIGN.md](DESIGN.md) for the architecture.

## Status

Pre-1.0. v1 ships the Django filter; the iommi-specific layer comes after.
Pinned against a narrow ty range — bumps are gated by a contract test
suite (`tests/test_contract_real_ty.py`).

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

**For the iommi analyzer**, run `iommi-lsp graph build` once after installing
or upgrading iommi in your project. The reflector imports iommi from the
same Python interpreter that runs `iommi-lsp` (i.e. install both in your
project's venv). The graph is a few hundred KB JSON in your workspace
root; check it in or `.gitignore` it as you prefer.

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

`iommi-lsp graph build` imports iommi in your venv and walks each
`Refinable`-declaring class (`Table`, `Column`, `Form`, `Field`, …)
transitively through `class_ref` and `members` edges. Every refinable
gets one of six kinds:

* `members` — open dict of typed values (`columns: Dict[str, Column]`)
* `html_attrs` — the `attrs` special with `class` (str→bool) and `style`
  (str→str) sub-namespaces
* `class_ref` — chain steps into another refinable class (annotation
  wins over runtime default, so `bulk: Optional[Form]` resolves to Form)
* `namespace` — structured with known sub-keys
* `open_namespace` — anything goes
* `evaluated_scalar` / `scalar` — leaf

At LSP time the analyzer finds every `Class(kw__chain=...)` call, splits
the kwarg name on `__`, and walks the chain through the graph. Dead
ends become `iommi-unknown-refinable` warnings pinned to the offending
segment. Bias is toward false negatives — if anything is ambiguous
(unknown root class, member with no typed value, custom user subclass
not in the graph), we pass silently.

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
If the receiver doesn't resolve, no items are offered. If `ty`
doesn't implement `textDocument/completion` we substitute our own
response so the editor still sees them; if it does, we merge.

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
* **No iommi awareness yet.** Coming after Django filtering stabilizes.
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
