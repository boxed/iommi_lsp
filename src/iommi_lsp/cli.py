"""``iommi_lsp`` entry point.

Two modes:

* No subcommand (default) — run as the LSP proxy on stdio. Spawns the
  bundled ``ty server`` (auto-detected next to our own interpreter)
  unless ``--ty-command`` overrides.
* ``iommi_lsp index <path>`` — build the Django model index for *path*
  and dump it to stdout. A debugging tool for milestone 3.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import __version__, log, proxy
from .analyzers.admin import AdminAnalyzer
from .analyzers.django import DjangoAnalyzer, build_index
from .analyzers.forms import FormsAnalyzer
from .analyzers.iommi import IommiAnalyzer
from .analyzers.iommi.build import GraphBuildError, build_for_workspace
from .analyzers.migrations import MigrationsAnalyzer
from .analyzers.settings import SettingsAnalyzer
from .analyzers.signals import SignalsAnalyzer
from .analyzers.templates import TemplateAnalyzer
from .analyzers.urls import UrlAnalyzer
from .analyzers.views import ViewsAnalyzer
from .parse_cache import ParseCache
from .interceptor import (
    CompletionMatchmaker,
    DefinitionRouter,
    DiagnosticInterceptor,
    DocumentStore,
    EditorRequestSniffer,
)


_log = log.get("cli")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iommi_lsp",
        description="Wrapper LSP that proxies ty and filters Django/iommi false positives.",
    )
    p.add_argument(
        "--ty-command",
        default=None,
        help="Command to spawn the backend type checker. Defaults to the "
             "bundled `ty server` next to this interpreter, falling back to "
             "`ty server` on PATH.",
    )
    p.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Eagerly index the given workspace at startup instead of waiting "
             "for the editor's `initialize` request.",
    )
    p.add_argument(
        "--log-level",
        default=None,
        help="Override the log level (DEBUG, INFO, WARNING, ERROR).",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"iommi_lsp {__version__}",
    )

    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    idx = sub.add_parser(
        "index",
        help="Build and print the Django model index for a workspace.",
    )
    idx.add_argument("path", type=Path, help="Workspace root to scan.")

    graph = sub.add_parser(
        "graph",
        help="Build / inspect the iommi reflection graph.",
    )
    graph_sub = graph.add_subparsers(dest="graph_command", metavar="ACTION")
    g_build = graph_sub.add_parser(
        "build",
        help="Reflect the installed iommi and write .iommi_lsp-graph.json.",
    )
    g_build.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Workspace root (default: cwd). Graph is written here.",
    )
    g_build.add_argument(
        "--python",
        default=None,
        help="Python interpreter to invoke (default: this venv's interpreter).",
    )
    g_build.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated list of fully-qualified iommi class seeds. "
             "Defaults to the public iommi exports.",
    )
    return p


def _resolve_ty_binary() -> str:
    """Locate the ``ty`` executable to spawn.

    Prefer the one shipped next to our own interpreter — when iommi_lsp
    is installed as a uv tool / pipx app, ``ty`` lives in the same
    ``bin/`` directory as ``sys.executable`` because we declare it as a
    hard dependency. Fall back to ``PATH``, then to the bare name (lets
    subprocess fail with a clear ENOENT if nothing's installed).
    """
    sibling_dir = str(Path(sys.executable).parent)
    return (
        shutil.which("ty", path=sibling_dir)
        or shutil.which("ty")
        or "ty"
    )


def _run_proxy(ty_command_str: str | None, workspace: Path | None) -> int:
    if ty_command_str is None:
        ty_command = [_resolve_ty_binary(), "server"]
    else:
        ty_command = shlex.split(ty_command_str)
    if not ty_command:
        print("error: --ty-command must not be empty", file=sys.stderr)
        return 2

    root = workspace or Path.cwd()
    documents = DocumentStore()
    # One AST per (uri, source) shared across analyzers. Without this,
    # ty's publishDiagnostics frame triggers each analyzer's
    # is_false_positive on every diagnostic; analyzers that re-parsed the
    # buffer per call multiplied that into seconds of dead time on
    # 10k-line files.
    parse_cache = ParseCache()
    parse_provider = parse_cache.get
    django_analyzer = DjangoAnalyzer(
        workspace_root=root, text_provider=documents.get,
        parse_provider=parse_provider,
    )
    iommi_analyzer = IommiAnalyzer(
        workspace_root=root,
        text_provider=documents.get,
        # Lets the iommi analyzer offer ``columns__<model_field>``
        # completions when the call carries ``auto__model=Model``.
        django_index_provider=lambda: django_analyzer.django_index,
        parse_provider=parse_provider,
    )
    url_analyzer = UrlAnalyzer(
        workspace_root=root, text_provider=documents.get,
        parse_provider=parse_provider,
    )
    template_analyzer = TemplateAnalyzer(
        workspace_root=root,
        text_provider=documents.get,
        # ``{% url %}`` completion + diagnostics in HTML templates draw
        # on the workspace URL index.
        url_index_provider=lambda: url_analyzer.url_index,
    )
    settings_analyzer = SettingsAnalyzer(
        workspace_root=root,
        text_provider=documents.get,
        # AUTH_USER_MODEL completion draws on workspace models.
        django_index_provider=lambda: django_analyzer.django_index,
    )
    admin_analyzer = AdminAnalyzer(
        workspace_root=root,
        text_provider=documents.get,
        django_index_provider=lambda: django_analyzer.django_index,
        parse_provider=parse_provider,
    )
    forms_analyzer = FormsAnalyzer(
        workspace_root=root,
        text_provider=documents.get,
        django_index_provider=lambda: django_analyzer.django_index,
        parse_provider=parse_provider,
    )
    views_analyzer = ViewsAnalyzer(
        workspace_root=root,
        text_provider=documents.get,
        django_index_provider=lambda: django_analyzer.django_index,
        parse_provider=parse_provider,
    )
    signals_analyzer = SignalsAnalyzer(
        workspace_root=root,
        text_provider=documents.get,
        django_index_provider=lambda: django_analyzer.django_index,
    )
    migrations_analyzer = MigrationsAnalyzer(
        workspace_root=root, text_provider=documents.get,
        parse_provider=parse_provider,
    )
    # Order matters for completion latency: ``_gather`` stops at the first
    # ``exclusive=True`` analyzer, so put the ones that are cheapest *and*
    # most likely to claim first.
    # - urls: ~0.2 ms (no ast.parse); bails immediately when the position
    #   isn't inside a ``reverse``-style call. Specific and very common,
    #   so first.
    # - templates: ~0.1 ms; bails immediately when the partial has no ``/``.
    # - settings: ~14 ms on a 1k-line buffer (it ast.parses the whole file
    #   to figure out the enclosing assignment), so it's gated to settings-
    #   style filenames internally — but we still place it after urls so
    #   ``reverse('|')`` in a settings file short-circuits before settings
    #   gets a chance to parse.
    # - iommi / django each cost ~0.7 ms in normal files because they parse
    #   the whole buffer to decide whether they own the position.
    analyzers = [
        url_analyzer, template_analyzer, settings_analyzer,
        admin_analyzer, forms_analyzer, views_analyzer,
        signals_analyzer, migrations_analyzer,
        iommi_analyzer, django_analyzer,
    ]

    interceptor = DiagnosticInterceptor(analyzers=analyzers)
    matchmaker = CompletionMatchmaker(
        analyzers=analyzers, text_provider=documents.get,
    )
    definition_router = DefinitionRouter(analyzers=analyzers)

    async def workspace_seen(root: Path) -> None:
        for a in analyzers:
            await a.index(root)

    async def file_changed(uri: str) -> None:
        for a in analyzers:
            await a.on_file_changed(uri)

    sniffer = EditorRequestSniffer(
        on_workspace=workspace_seen,
        on_file_changed=file_changed,
        document_store=documents,
    )

    editor_to_ty = _chain_hooks(
        sniffer, matchmaker.on_request, definition_router.on_request,
    )
    ty_to_editor = _chain_hooks(
        matchmaker.on_response, definition_router.on_response, interceptor,
    )

    backend_env = _backend_env(root, os.environ)

    def attach_writer(writer) -> None:
        matchmaker.attach_editor_writer(writer)
        definition_router.attach_editor_writer(writer)

    if workspace is not None:
        return asyncio.run(_eager_index_then_serve(
            ty_command, analyzers, workspace, editor_to_ty, ty_to_editor, backend_env,
            on_writer_ready=attach_writer,
        ))
    return asyncio.run(proxy.run(
        ty_command,
        editor_to_ty_hook=editor_to_ty,
        ty_to_editor_hook=ty_to_editor,
        env=backend_env,
        on_writer_ready=attach_writer,
    ))


def _chain_hooks(*hooks):
    """Compose proxy hooks left-to-right. Each hook receives the bytes
    produced by the previous; ``None`` from any link short-circuits."""
    async def call(body: bytes) -> bytes | None:
        for h in hooks:
            if body is None:
                return None
            body = await h(body)
        return body
    return call


async def _eager_index_then_serve(
    ty_command, analyzers, workspace, editor_to_ty_hook, ty_to_editor_hook, env,
    *, on_writer_ready=None,
) -> int:
    for a in analyzers:
        await a.index(workspace)
    return await proxy.run(
        ty_command,
        editor_to_ty_hook=editor_to_ty_hook,
        ty_to_editor_hook=ty_to_editor_hook,
        env=env,
        on_writer_ready=on_writer_ready,
    )


def _backend_env(
    workspace: Path, current_env: Mapping[str, str]
) -> Mapping[str, str]:
    """Build the env for the backend (ty) subprocess.

    Two adjustments vs. inheriting the parent env:

    1. ``VIRTUAL_ENV`` (and ``PATH``) is set to the workspace's ``.venv``
       (or ``venv``) when present and the parent env doesn't already have
       a valid ``VIRTUAL_ENV``. Editors that launch iommi_lsp via wrapper
       scripts (uv tool, pipx) often strip or overwrite ``VIRTUAL_ENV``,
       leaving ty unable to find the workspace's installed packages.
    2. The directory holding this Python interpreter is *appended* to
       ``PATH``. When iommi_lsp is installed as a uv tool, the bundled
       ``ty`` lives next to ``sys.executable`` but isn't on the editor's
       PATH; this lets us find it as a last resort.
    """
    new_env = dict(current_env)

    existing = current_env.get("VIRTUAL_ENV")
    if existing and (Path(existing) / "bin" / "python").exists():
        _log.info("inherited VIRTUAL_ENV=%s", existing)
    else:
        for candidate_name in (".venv", "venv"):
            candidate = workspace / candidate_name
            if (candidate / "bin" / "python").exists():
                new_env["VIRTUAL_ENV"] = str(candidate)
                new_env["PATH"] = str(candidate / "bin") + os.pathsep + new_env.get("PATH", "")
                new_env.pop("PYTHONHOME", None)
                _log.info("injecting VIRTUAL_ENV=%s for backend", candidate)
                break
        else:
            _log.info("no venv found at %s/.venv or %s/venv", workspace, workspace)

    own_bin = str(Path(sys.executable).parent)
    path_parts = new_env.get("PATH", "").split(os.pathsep)
    if own_bin not in path_parts:
        new_env["PATH"] = (new_env.get("PATH", "") + os.pathsep + own_bin).lstrip(os.pathsep)
        _log.info("appended %s to PATH (bundled-backend fallback)", own_bin)

    return new_env


def _run_graph_build(path: Path, python: str | None, seeds: str | None) -> int:
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2
    kwargs: dict = {"python": python}
    if seeds:
        kwargs["seeds"] = tuple(s.strip() for s in seeds.split(",") if s.strip())
    try:
        out = build_for_workspace(path, **kwargs)
    except GraphBuildError as e:
        print(f"error: graph build failed: {e}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


def _run_index(path: Path) -> int:
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2
    index = build_index(path)
    print(index.summary())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    log.configure(level=args.log_level)

    try:
        if args.command == "index":
            return _run_index(args.path)
        if args.command == "graph":
            if args.graph_command == "build":
                return _run_graph_build(args.path, args.python, args.seeds)
            print("usage: iommi_lsp graph build [path]", file=sys.stderr)
            return 2
        return _run_proxy(args.ty_command, args.workspace)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
