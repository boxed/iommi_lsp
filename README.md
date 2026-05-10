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
iommi-lsp index ./myproject                  # dump the model index and exit
```

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

## How the filter works

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
`reverse`. Unknown groups in `disabled_rules` are ignored with a stderr
warning rather than silently breaking the filter.

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
