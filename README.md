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
  `prefetch_related` strings, `Q(...)` / `F('...')` expressions,
  `Count` / `Sum` / `Avg` / `Min` / `Max` / `OuterRef` / `Subquery`
  string args inside `annotate(...)` / `aggregate(...)`,
  `Prefetch('rel', queryset=...)` calls, and the
  `get_object_or_404(Model, name=...)` / `get_list_or_404(...)`
  shortcuts — full `__`-traversal through relations and reverse
  relations. Aliases declared by a sibling `.annotate(name=...)` /
  `.aggregate(name=...)` / `.alias(name=...)` in the same expression
  chain are accepted as valid leaves on downstream `.filter()` /
  `.order_by()` calls.

  ![](docs/screenshots/out/orm-diagnostic.png)

* **URL-name awareness.** A workspace index of `urls.py` files
  collects every `name='…'` from `path()` / `re_path()` / `url()`,
  honours `include(...)` namespaces (and `app_name = '…'`), and feeds:
  completion inside `reverse('‸')` / `reverse_lazy('‸')` /
  `redirect('‸')` / `resolve_url('‸')`, plus
  `django-unknown-url-name` diagnostics for typos. The same index is
  reused inside Django templates: `{% url '‸' %}` completes against
  the index, and unknown names raise `django-unknown-url-name`.

  ![](docs/screenshots/out/url-completion.png)
  ![](docs/screenshots/out/template-url-tag.png)
  ![](docs/screenshots/out/template-url-diagnostic.png)

* **Admin field validation.** Inside `class FooAdmin(admin.ModelAdmin):`
  (registered via `@admin.register(MyModel)` or `admin.site.register(MyModel, FooAdmin)`),
  the entries of `list_display`, `list_filter`, `search_fields`,
  `readonly_fields`, `ordering`, `autocomplete_fields`, `fields`,
  `exclude`, `fieldsets` (nested `'fields'` key), and
  `prepopulated_fields` are checked against the model — `'eemail'`
  flags as `django-unknown-admin-field`, and completion offers the
  real field names. Sigils (`-` for ordering, `=` / `^` / `@` for
  search_fields) are handled transparently.

  ![](docs/screenshots/out/admin-completion.png)

* **ModelForm / Form awareness.** `Meta.fields` / `Meta.exclude` are
  validated against the bound model; `Meta.widgets` / `Meta.labels` /
  `Meta.help_texts` / `Meta.error_messages` / `Meta.field_classes`
  dict keys are validated the same way and complete against model
  fields; `clean_<field>` methods on `Form` / `ModelForm` subclasses
  fire `django-unknown-clean-method` when `<field>` isn't a declared
  form field; completion fires inside `self.fields['‸']` /
  `self.cleaned_data['‸']`.

  ![](docs/screenshots/out/forms-completion.png)
  ![](docs/screenshots/out/forms-meta-dict.png)

* **Class-based view attributes.** `model = Foo` binds the CBV; then
  `fields = ['‸']` (UpdateView/CreateView), `ordering = ['-‸']`
  (ListView), and `slug_field = '‸'` (DetailView) all complete and
  validate against `Foo`. Inherited mixin attrs (`paginate_by`,
  `context_object_name`, `slug_url_kwarg`, `pk_url_kwarg`,
  `template_name`, `queryset`, `form_class`, `success_url`, …) accessed
  as `self.<attr>` in a CBV subclass no longer trip ty's
  `unresolved-attribute` warning — those are real Django API, just
  invisible to ty without runtime stubs.

  ![](docs/screenshots/out/views-completion.png)

* **Migration dependencies.** `dependencies = [('app', '‸')]` in a
  `Migration` subclass offers the matching `<app>/migrations/`
  filenames. `RunPython.noop` / `RunSQL.noop` (passed as the reverse
  operation in a data migration) no longer trip
  `unresolved-attribute` either.

* **Signal sender completion.** `@receiver(post_save, sender=‸)` and
  `signal.connect(handler, sender=‸)` surface workspace model
  classes; the first positional of `@receiver(...)` and `signal=`
  kwargs suggest known signal names (`post_save`, `pre_delete`, …).

  ![](docs/screenshots/out/signal-completion.png)

* **Staticfiles completion.** Typing inside `static('‸')` (Python) or
  `{% static '‸' %}` (template) offers every file under any `static/`
  directory in the project.

  ![](docs/screenshots/out/static-completion.png)
  ![](docs/screenshots/out/template-static-tag.png)

* **Template-name completion in any `/`-containing string.** Once the
  workspace is indexed, typing `'myapp/‸'` (where `‸` is the cursor)
  offers every file under any `templates/` directory in the project.
  Non-exclusive: ty's path-style suggestions still come through so
  non-template paths aren't suppressed.

  ![](docs/screenshots/out/template-completion.png)

* **Django template-tag awareness.** Inside `.html` files, the LSP
  recognises Django's tag syntax and completes contextually:
  `{% extends '‸' %}` / `{% include '‸' %}` offer template names with
  no `/` heuristic (the tag itself is unambiguous);
  `{% block ‸ %}` in a child template reads the parent's
  `{% extends '...' %}` and surfaces the parent's block names (with
  one level of grandparent recursion); `{% load ‸ %}` autocompletes
  any `templatetags/` library discovered across the workspace.

  ![](docs/screenshots/out/template-extends-tag.png)
  ![](docs/screenshots/out/template-block-tag.png)
  ![](docs/screenshots/out/template-load-tag.png)

* **Template filter completion.** After a `|` inside `{{ ‸ }}` or
  `{% if ‸ %}`, the popup offers every filter in
  `django.template.defaultfilters` (built-in — always available) plus
  every `@register.filter` discovered in any `templatetags/` library
  the current template `{% load %}`s. Custom filters surface their
  registered name (so `@register.filter(name='renamed')` shows up as
  `renamed`, not the function name). Library filters from libraries
  the file hasn't loaded are filtered out.

  ![](docs/screenshots/out/template-filter.png)

* **`INSTALLED_APPS` / settings completion.** Inside `INSTALLED_APPS`,
  `MIDDLEWARE`, `AUTHENTICATION_BACKENDS`, `AUTH_USER_MODEL`,
  `DEFAULT_AUTO_FIELD`, `WSGI_APPLICATION`,
  `AUTH_PASSWORD_VALIDATORS`, and `DEFAULT_EXCEPTION_REPORTER` string
  values, you get the appropriate set of dotted-path suggestions —
  Django's contribs and middleware, workspace apps (any package with
  `apps.py`), workspace models in `app_label.ModelName` form, etc.

  ![](docs/screenshots/out/settings-installed-apps.png)

* **No more `Item.objects` false positives.** ty's
  `unresolved-attribute` diagnostics on Django metaclass magic
  (`objects`, `_meta`, `pk`/`id`, `<fk>_id`, reverse relations,
  `DoesNotExist`, `MultipleObjectsReturned`,
  `get_<field>_display()` for fields with `choices=`,
  `get_next_by_<field>()` / `get_previous_by_<field>()` on date
  fields, `<m2m>.through` on `ManyToManyField` descriptors, …) are
  dropped before they reach the editor. Real bugs survive. Custom
  manager methods surfaced via `objects = MyQuerySet.as_manager()` or
  a `models.Manager` subclass are picked up workspace-wide so
  `Order.objects.<custom_method>()` stops nagging too.

* **`get_user_model()` awareness.** `UserCls = get_user_model();
  UserCls.objects.filter(...)` resolves the binding to the contrib
  `User` model (or to a workspace `User` that shadows it), so kwargs
  and field-path strings validate against the right schema.
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
  literals, and also against top-level `model=` / `rows=` /
  `instance=` kwargs so `Table(rows=User.objects.all(), columns__‸)`
  works without `auto__`.

  ![](docs/screenshots/out/iommi-auto-model.png)

* **Style completion.** `Table(style='‸')` / `Form(style='‸')`
  offers iommi's built-in style names (`bootstrap`, `bootstrap5`,
  `bulma`, `water`, …). Non-exclusive so custom-registered styles
  still come through ty.

* **`iommi-unknown-refinable` diagnostics.** Invalid chains in
  `Class(kw__chain=...)` calls flag the first dead-end segment.

* **`attr=` value bridging.** When `auto__model=` / `rows=` / `model=` /
  `instance=` is in scope, the string value of
  `fields__name__attr='nested__path'` (Form) or
  `columns__name__attr='nested__path'` (Table) is validated as a Django
  model lookup against the bound model — `iommi-unknown-attr-path`
  flags the first dead segment, pinned to that segment in the string.

  ![](docs/screenshots/out/iommi-attr-bridge.png)

* **`iommi-callable-expected` diagnostics.** A string literal at a
  callable-expecting refinable — `Action(post_handler='save')`,
  `Form(endpoints__name__func='view')`, `Form(on_save='handler')`,
  `Form(on_commit='c')` — gets flagged. Almost always a typo where
  the user meant a name reference and accidentally quoted it.

  ![](docs/screenshots/out/iommi-callable.png)
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
      filetypes = { "python", "htmldjango", "html" },
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

[[language]]
name = "html"
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
    },
    "HTML": {
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
`reverse`, `generated`, `orm_lookup`, `unused_request_param`. Unknown
groups in `disabled_rules` are ignored with a stderr warning rather
than silently breaking the filter.

A missing or malformed `pyproject.toml` falls back to defaults; the
proxy never crashes on a bad config.

## Development

```sh
uv venv
uv pip install -e ".[dev]"
uv run pytest
```
