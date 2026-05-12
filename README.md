# iommi_lsp

A Django and iommi language server that proxies to
[`ty`](https://github.com/astral-sh/ty) for broad Python support.

## What you get

### Django

* **Real autocomplete for ORM kwargs.** Inside `User.objects.filter(‸`,
  `.exclude(‸`, `.get(‸`, `.update(‸`, `.create(‸`, `.get_or_create(‸`,
  `.update_or_create(‸` (where `‸` is the cursor) you get the model's queryable names — declared
  fields, FK `_id` accessors, `pk`, reverse-relation accessors —
  with `__`-traversal into related models. Suggestions insert as
  `name=` so the caret lands at the value. At a recognised call site
  we claim **exclusivity**, so ty's "any local variable near `em`"
  noise stays out of the list.

  ![](docs/screenshots/out/orm-completion.png)

* **Typo diagnostics ty can't see.** `django-unknown-orm-lookup`
  warnings fire on kwargs and string field paths that don't resolve
  against the workspace model index. Covers `.filter(...)` /
  `.exclude(...)` / `.get(...)` and friends, `order_by` / `values` /
  `values_list` / `only` / `defer` / `distinct` / `select_related` /
  `prefetch_related` strings, `Q(...)` / `F('...')` expressions —
  full `__`-traversal through relations and reverse relations.

  ![](docs/screenshots/out/orm-diagnostic.png)

* **No more `Item.objects` false positives.** ty's
  `unresolved-attribute` diagnostics on Django metaclass magic
  (`objects`, `_meta`, `pk`/`id`, `<fk>_id`, reverse relations,
  `DoesNotExist`, `MultipleObjectsReturned`, …) are dropped before they
  reach the editor. Real
  bugs survive.
* **No more `` `request` is unused `` nags on views.** Django view
  functions take `request` whether they read it or not — ty's hint is
  dropped when `request` is the first parameter (or first after
  `self`/`cls` on a class-based view). Other unused params still flag,
  and an unused *local* `request` variable still flags.
* **Built-in models + abstract inheritance.** `django.contrib.auth`
  / `contenttypes` / `sessions` models are stubbed so they work out
  of the box, and abstract-base fields propagate to concrete
  subclasses (so a custom `User(AbstractUser)` resolves `email` /
  `username` / etc.).

### iommi

* **Refinable autocomplete inside `Class(kw__chain=...)` calls.**
  `Table(c‸` (where `‸` is the cursor) suggests `columns__`, `cell__`, `query__`, …;
  containers get a trailing `__` and scalars get `=`. Chains walk
  the iommi refinable graph, so `Table(columns__name__‸` offers
  the configurable surface of `Column`.

  ![](docs/screenshots/out/iommi-refinable.png)

* **`auto__` namespace.** Always surfaces `model` / `rows` /
  `instance` / `include` / `exclude` whether or not the graph
  reflects it, since iommi's default `Namespace()` is empty.
* **Django field bridging.** `Table(auto__model=User, columns__‸)`
  suggests `User`'s fields (insert as `username__`, `email__`, …
  so you can keep configuring the auto-generated column). The same
  works inside `auto__include=['‸']` / `auto__exclude=['‸']` string
  literals.

  ![](docs/screenshots/out/iommi-auto-model.png)

* **`iommi-unknown-refinable` diagnostics.** Invalid chains in
  `Class(kw__chain=...)` calls flag the first dead-end segment.
* **Zero-setup defaults.** Synthesised stubs cover the public iommi
  classes (`Table`, `Form`, `Query`, `Page`) so all of the above
  works before any graph build succeeds; the project's own iommi
  subclasses light up once a real graph is built (automatically, in
  most setups — see below).

It speaks plain LSP, runs on stdio, and is configured into your editor
in place of `ty server`. See [ARCHITECTURE.md](ARCHITECTURE.md) for how
the analyzers work internally, and [DESIGN.md](DESIGN.md) for the
higher-level design rationale.

## Status

Pre-1.0. Pinned against a narrow ty range — bumps are gated by a
contract test suite (`tests/test_contract_real_ty.py`).

## Install

```sh
uv tool install iommi_lsp     # or: pipx install iommi_lsp
```

`ty` is a hard dependency and is installed alongside `iommi_lsp` into
the same environment, so the default just works — no editor-side
`--ty-command` plumbing required.

## Run

```sh
iommi_lsp                                    # spawns the bundled `ty server`
iommi_lsp --ty-command "uvx ty server"       # override (e.g. pin a different ty)
iommi_lsp --workspace ./myproject            # eager indexing for debugging
iommi_lsp index ./myproject                  # dump the Django model index and exit
iommi_lsp graph build ./myproject            # reflect installed iommi -> .iommi_lsp-graph.json
```

**For the iommi analyzer**, the graph at `.iommi_lsp-graph.json` is built
automatically when the workspace is opened:

1. **In-process** if `iommi` is importable from `iommi_lsp`'s interpreter
   (i.e. installed alongside it: `uv tool install --with iommi iommi_lsp`).
2. **Subprocess** against the workspace's `.venv` / `venv` Python, when
   `iommi_lsp` is installed there too.
3. **Synthesized stubs** for the well-known iommi classes (`Table`,
   `Form`, `Query`, `Page`) as a last resort — enough that `auto__…` and
   members-name completion still work before any graph build succeeds.

Running `iommi_lsp graph build` by hand is still supported and is the
fastest way to force a rebuild after upgrading iommi. The graph is a few
hundred KB JSON in your workspace root; check it in or `.gitignore` it
as you prefer.

`iommi_lsp` writes diagnostics-side stderr logs; tune via
`IOMMI_LSP_LOG=DEBUG` or `--log-level DEBUG`.

## Editor configuration

### Neovim (`nvim-lspconfig` style)

```lua
local lspconfig = require("lspconfig")
local configs = require("lspconfig.configs")

if not configs.iommi_lsp then
  configs.iommi_lsp = {
    default_config = {
      cmd = { "iommi_lsp" },
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
[language-server.iommi_lsp]
command = "iommi_lsp"

[[language]]
name = "python"
language-servers = ["iommi_lsp"]
```

### Zed (`settings.json` under `lsp`)

```json
{
  "lsp": {
    "iommi_lsp": {
      "binary": { "path": "iommi_lsp" }
    }
  },
  "languages": {
    "Python": {
      "language_servers": ["iommi_lsp"]
    }
  }
}
```

### VS Code

There's no first-party VS Code extension yet. The simplest path is the
[`vscode-generic-lsp-client`](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
pattern: install a generic LSP-client extension and point it at
`iommi_lsp`. A first-party extension is on the roadmap.

## Per-project configuration

Add a `[tool.iommi_lsp]` table to your `pyproject.toml`:

```toml
[tool.iommi_lsp]
enabled = true                          # master switch
disabled_rules = ["pk", "reverse"]       # skip rule groups for this project

[tool.iommi_lsp.extra_magic_attrs]
manager = ["mongo", "search"]            # treat these as Manager-like attrs
```

Recognised rule groups: `manager`, `meta`, `pk`, `exception`, `fk_id`,
`reverse`, `orm_lookup`, `unused_request_param`. Unknown groups in
`disabled_rules` are ignored with a stderr warning rather than silently
breaking the filter.

A missing or malformed `pyproject.toml` falls back to defaults; the
proxy never crashes on a bad config.

## Caveats

* **Pre-1.0 ty.** Diagnostic codes and message text *will* change. The
  contract suite (`tests/test_contract_real_ty.py`) catches breakage when
  you bump ty.
* **iommi graph requires iommi to be importable somewhere.** Either in
  the same venv as `iommi_lsp`, or in the workspace's `.venv` / `venv`.
  Without that, the synthesised stubs cover the public iommi classes
  but project-specific subclasses (and their refinables) stay invisible
  until you run `iommi_lsp graph build`.
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
