# iommi-lsp — Design Document

A wrapper Language Server that proxies [`ty`](https://github.com/astral-sh/ty)
and filters out false-positive diagnostics caused by Django's metaclass magic.
Eventually grows iommi-specific awareness; **v1 is Django filtering only**.

The name reflects the long-term goal. Django support is the necessary
precursor since iommi sits on top of Django.

---

## 1. Problem

`ty` is fast, correct, and editor-agnostic, but [does not support a plugin
system](https://saurabh-kumar.com/articles/2025/05/thoughts-on-astrals-ty-the-lightning-fast-python-type-checker-language-server/)
("Astral sees it as a feature that type checking works interchangeably across
tools and projects"). Django (and iommi) rely heavily on metaclass-injected
attributes and runtime-computed reverse relations that pure stub files cannot
fully express. The result: thousands of `unresolved-attribute` false positives
on a real Django codebase.

Running `ty` and a Django-aware LSP (e.g. `djls`) in parallel does **not**
solve this — LSP diagnostics from multiple servers are additive, not
consensus-based. One server saying "fine" never suppresses another saying
"error."

## 2. Approach

Single LSP that the editor talks to. It spawns `ty server` as a subprocess
and proxies JSON-RPC messages bidirectionally. Most messages pass through
verbatim. `textDocument/publishDiagnostics` going ty→editor is intercepted,
filtered through framework-aware analyzers, and forwarded.

```
            ┌────────┐  LSP  ┌──────────────┐  LSP  ┌────────┐
   editor ──┤ client ├───────┤  iommi-lsp   ├───────┤   ty   │
            └────────┘       │   (proxy)    │       └────────┘
                             │              │
                             │  ┌────────┐  │
                             │  │analyzer│  │  ← v1: Django
                             │  │ (incl. │  │  ← later: iommi
                             │  │ index) │  │
                             │  └────────┘  │
                             └──────────────┘
```

### Why proxy and not parallel server

- **Diagnostics are subtractive in our case** (we want to remove ty's false
  positives, not add new ones). The only place to subtract is between ty and
  the editor.
- One server end-to-end means no duplicate hover/completion noise later when
  we add augmentation features.
- The analyzer becomes a reusable layer for other type checkers later
  (mypy, pyright). The design treats `ty` as just the first backend.

## 3. Architecture

### 3.1 Components

```
iommi-lsp
├── proxy        — JSON-RPC framing, subprocess plumbing, message pumping
├── interceptor  — picks messages to inspect/rewrite (just diagnostics in v1)
├── analyzers/
│   └── django   — workspace index + false-positive predicates
└── index        — generic AST-based workspace introspection helpers
```

The analyzer interface is defined upfront even though only Django ships in v1
— this is what lets iommi plug in later without reshaping the proxy.

### 3.2 Analyzer interface

```python
class Analyzer(Protocol):
    name: str

    async def index(self, workspace_root: Path) -> None:
        """Build/refresh the analyzer's view of the workspace."""

    async def on_file_changed(self, uri: str) -> None:
        """Update the index for a single file."""

    def is_false_positive(
        self,
        uri: str,
        diagnostic: Diagnostic,
    ) -> bool:
        """Return True if this diagnostic should be dropped."""
```

The proxy holds a list of analyzers and drops a diagnostic if **any** analyzer
flags it. This keeps the contract simple and makes it trivial to add iommi
later as a second analyzer.

### 3.3 Message flow

1. Editor sends `initialize` → proxy forwards to ty → ty replies → proxy
   forwards reply (possibly merging in extra capabilities later, none in v1).
2. Editor sends `textDocument/didOpen` → proxy forwards to ty AND notifies
   analyzers (for index updates).
3. ty publishes diagnostics → proxy filters via analyzers → forwards reduced
   list.
4. Shutdown is symmetric: editor sends shutdown/exit, proxy forwards, awaits
   ty exit, exits itself.

## 4. Tech stack

| Concern              | Choice                | Rationale |
| -------------------- | --------------------- | --------- |
| Language             | Python 3.11+          | Same ecosystem as Django/iommi; lets the analyzer import the project |
| Async runtime        | `asyncio`             | Standard, sufficient for stdio pumping |
| LSP message framing  | Hand-rolled (~50 LOC) | We're proxying, not implementing a server; `pygls` would be overkill |
| LSP types            | `lsprotocol`          | Just for the diagnostic and message dataclasses |
| AST                  | `libcst` or `ast`     | Start with stdlib `ast`; switch to `libcst` only if we need round-trip fidelity |
| Subprocess mgmt      | `asyncio.subprocess`  | |
| Packaging            | `hatchling` + `uv`    | uvx-friendly entry point |
| Testing              | `pytest` + corpus     | Recorded LSP traces + small Django project fixtures |

**Pinning ty.** ty is pre-1.0; rule names and message formats can change.
Pin a specific ty version range in `pyproject.toml` and add a contract test
suite (see §7).

## 5. Project layout

```
iommi-lsp/
├── pyproject.toml
├── README.md
├── DESIGN.md                       # this file
├── src/iommi_lsp/
│   ├── __init__.py
│   ├── __main__.py                 # `python -m iommi_lsp`
│   ├── cli.py                      # entry point: `iommi-lsp`
│   ├── proxy.py                    # main proxy server
│   ├── jsonrpc.py                  # Content-Length framing
│   ├── interceptor.py              # diagnostic filtering pipeline
│   ├── analyzers/
│   │   ├── __init__.py
│   │   ├── base.py                 # Analyzer protocol + registry
│   │   └── django/
│   │       ├── __init__.py
│   │       ├── analyzer.py
│   │       ├── index.py            # model registry, related_name graph
│   │       └── magic.py            # known Django attrs (objects, _meta, …)
│   └── log.py                      # structured logging to stderr
├── tests/
│   ├── conftest.py
│   ├── corpus/
│   │   ├── basic_django/           # tiny Django project fixture
│   │   └── related_names/          # FK/M2M reverse-relation fixture
│   ├── test_jsonrpc.py
│   ├── test_proxy.py               # spawns dummy ty subprocess
│   ├── test_django_index.py
│   └── test_django_filter.py       # the contract test suite vs real ty
└── .github/workflows/test.yml
```

## 6. Django analyzer (v1 scope)

### 6.1 What we filter

Only `unresolved-attribute` diagnostics, and only when:

1. The diagnostic's range falls on an attribute access `X.attr`.
2. `X`'s static type resolves to a subclass of `django.db.models.Model`
   (transitively, through abstract bases).
3. `attr` is one of:
   - **Manager-like**: `objects`, `_default_manager`, `_base_manager`
   - **Meta**: `_meta`, `Meta`
   - **PK aliases**: `pk`, `id` (when no explicit PK is declared)
   - **Exception classes**: `DoesNotExist`, `MultipleObjectsReturned`
   - **Reverse relations**: any name reachable in the workspace's
     related-name graph (see §6.2)
   - **FK ID accessors**: `<fk_field>_id` for any declared `ForeignKey` or
     `OneToOneField`

Anything outside this set is forwarded unchanged. **Bias toward false
negatives** (let some noise through) rather than false positives (suppress
real bugs).

### 6.2 Workspace index

On first use and on file change, walk the project's Python files (respecting
the workspace root from `initialize`) and build:

```python
@dataclass
class DjangoIndex:
    models: dict[str, ModelInfo]              # qualname -> info
    reverse_relations: dict[str, set[str]]    # model qualname -> attr names
```

`ModelInfo` records declared fields, `Meta`, and PK declaration. The reverse
relations map is computed from `ForeignKey`/`OneToOneField`/`ManyToManyField`
calls — `related_name=` if present, otherwise default `<lowermodel>_set`.

The index is **AST-only** — we do not import the user's code. Importing
Django settings is the kind of fragility we explicitly rejected in §1.

### 6.3 Locating the receiver type

The diagnostic gives us the byte range of the offending expression. To answer
"is `X` a Django model?" we need to know the type of `X`, which `ty` knows but
hasn't told us. Three options, in increasing fidelity:

- **a. Syntactic match.** Parse the line, look at the receiver. If it's a
  bare name matching a known model class, treat it as that model. Cheap, gets
  ~70% of cases (`User.objects`, `Article.objects`).
- **b. Local flow.** AST-walk the enclosing function for assignments
  (`user = User.objects.get(...)`) and follow the type. Gets ~95%.
- **c. Ask ty.** Send a `textDocument/hover` request back to ty for the
  receiver position and parse the type. Most accurate but adds a round-trip.

**v1 ships (a)+(b).** (c) is a fallback we add only if the corpus tests
show we need it.

## 7. Contract tests

A directory of `.py` fixtures, each annotated with expected post-filter
diagnostics. The test harness:

1. Boots the real `ty server` as a subprocess.
2. Boots `iommi-lsp` pointing at the fixture directory.
3. Sends `didOpen` for each fixture.
4. Collects published diagnostics.
5. Compares against expected.

This is the suite that catches breakage when `ty` is bumped. Run it in CI
against the pinned ty version *and* the latest, with the latest allowed to
fail.

## 8. Milestones for v1

Each milestone ends in a working, demoable artifact. Implement in order.

1. **Skeleton + echo proxy.** `pyproject.toml`, package structure, `iommi-lsp`
   entry point. Spawns `ty server`, pumps JSON-RPC both directions verbatim.
   Demo: editor connects via `iommi-lsp`, behaves identically to direct ty.

2. **Diagnostic interception.** Parse `textDocument/publishDiagnostics`,
   log to stderr, forward unchanged. Confirms we can read structured
   diagnostics without breaking the protocol.

3. **Django model index.** AST scan of workspace, builds `DjangoIndex`.
   Standalone CLI for debugging: `iommi-lsp index <path>` prints the index.
   No proxy integration yet.

4. **Magic-attribute filter.** Wire the index into the interceptor. Drop
   `unresolved-attribute` diagnostics where receiver is a known model and
   attribute is in the static magic set (`objects`, `_meta`, `pk`, ...).

5. **Reverse relations.** Compute the `related_name` graph during indexing.
   Extend the filter to drop reverse-relation false positives.

6. **FK ID accessors.** Add `<field>_id` recognition for `ForeignKey`/
   `OneToOneField`.

7. **File-change handling.** `didChange`/`didSave` updates the index
   incrementally.

8. **Packaging + release.** Publish to PyPI as `iommi-lsp`, runnable via
   `uvx iommi-lsp`. README with editor-config snippets for Neovim, Helix,
   Zed, VS Code.

## 9. Future (out of scope for v1)

- **iommi analyzer.** Index iommi `Page`/`Form`/`Table` declarations and
  their refinables. Likely needs runtime introspection (importing the
  project) since iommi's structure is computed, unlike Django models.
  Optional opt-in flag: `--allow-import` or similar.
- **Augmentation.** Hover enrichment, reverse-relation completions,
  Django-specific diagnostics (e.g. unknown field names in `.filter()`),
  code actions ("add `if TYPE_CHECKING: ...`").
- **Other backends.** Allow proxying mypy or pyright in place of ty.
- **Configuration file.** `pyproject.toml`'s `[tool.iommi-lsp]` for
  per-project overrides — disabled rules, custom magic attrs, etc.

## 10. Caveats and explicit non-goals

- **Pre-1.0 ty.** Diagnostic codes and message text *will* change. Pin the
  version, run the contract suite in CI.
- **Performance.** Every LSP message round-trips through this proxy. Filter
  work must stay off the message-pump task — push it to a worker if it ever
  takes longer than a few ms.
- **Indexing cost.** First-pass workspace scan is on the proxy's startup
  path. Make it lazy: build the index in the background after `initialize`,
  and let early diagnostics through unfiltered until it's ready.
- **No runtime introspection in v1.** Pure AST. Importing user code is a
  fragility surface we don't take on yet.
- **Scope creep risk.** This will start to look like
  `mypy_django_plugin`. That's fine — but resist building stubs. Stubs
  belong in `django-types`; we just consume them transparently via ty.
- **Upstream risk.** Astral has signaled interest in first-class library
  support inside ty. If they ship Django awareness, v1's Django filter
  becomes mostly redundant. The iommi layer remains valuable, and the
  proxy architecture remains useful as a place to put project-specific
  rules. Build accordingly.

## 11. Getting started for the implementer

Recommended first session:

1. Skim §§1–3 for context.
2. Implement milestone 1 (skeleton + echo proxy) end to end.
3. Write a single integration test that confirms a real editor (or
   `lsp-devtools` proxy) sees identical behavior with iommi-lsp vs.
   direct ty.
4. Stop, commit, demo.
5. Then milestone 2.

Resist the urge to scaffold all milestones at once. Each one should land
green.