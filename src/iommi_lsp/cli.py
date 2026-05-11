"""``iommi-lsp`` entry point.

Two modes:

* No subcommand (default) — run as the LSP proxy on stdio. Spawns
  ``ty server`` from ``PATH`` unless ``--ty-command`` overrides.
* ``iommi-lsp index <path>`` — build the Django model index for *path*
  and dump it to stdout. A debugging tool for milestone 3.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import __version__, log, proxy
from .analyzers.django import DjangoAnalyzer, build_index
from .analyzers.iommi import IommiAnalyzer
from .analyzers.iommi.build import GraphBuildError, build_for_workspace
from .interceptor import (
    CompletionMatchmaker,
    DiagnosticInterceptor,
    DocumentStore,
    EditorRequestSniffer,
)


_log = log.get("cli")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iommi-lsp",
        description="Wrapper LSP that proxies ty and filters Django/iommi false positives.",
    )
    p.add_argument(
        "--ty-command",
        default="ty server",
        help="Command to spawn the backend type checker (default: %(default)r).",
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
        version=f"iommi-lsp {__version__}",
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
        help="Reflect the installed iommi and write .iommi-lsp-graph.json.",
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


def _run_proxy(ty_command_str: str, workspace: Path | None) -> int:
    ty_command = shlex.split(ty_command_str)
    if not ty_command:
        print("error: --ty-command must not be empty", file=sys.stderr)
        return 2

    root = workspace or Path.cwd()
    documents = DocumentStore()
    django_analyzer = DjangoAnalyzer(
        workspace_root=root, text_provider=documents.get,
    )
    iommi_analyzer = IommiAnalyzer(
        workspace_root=root, text_provider=documents.get,
    )
    analyzers = [django_analyzer, iommi_analyzer]

    interceptor = DiagnosticInterceptor(analyzers=analyzers)
    matchmaker = CompletionMatchmaker(analyzers=analyzers)

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

    editor_to_ty = _chain_hooks(sniffer, matchmaker.on_request)
    ty_to_editor = _chain_hooks(matchmaker.on_response, interceptor)

    backend_env = _backend_env(root, os.environ)

    if workspace is not None:
        return asyncio.run(_eager_index_then_serve(
            ty_command, analyzers, workspace, editor_to_ty, ty_to_editor, backend_env,
        ))
    return asyncio.run(proxy.run(
        ty_command,
        editor_to_ty_hook=editor_to_ty,
        ty_to_editor_hook=ty_to_editor,
        env=backend_env,
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
) -> int:
    for a in analyzers:
        await a.index(workspace)
    return await proxy.run(
        ty_command,
        editor_to_ty_hook=editor_to_ty_hook,
        ty_to_editor_hook=ty_to_editor_hook,
        env=env,
    )


def _backend_env(
    workspace: Path, current_env: Mapping[str, str]
) -> Mapping[str, str]:
    """Build the env for the backend (ty) subprocess.

    Two adjustments vs. inheriting the parent env:

    1. ``VIRTUAL_ENV`` (and ``PATH``) is set to the workspace's ``.venv``
       (or ``venv``) when present and the parent env doesn't already have
       a valid ``VIRTUAL_ENV``. Editors that launch iommi-lsp via wrapper
       scripts (uv tool, pipx) often strip or overwrite ``VIRTUAL_ENV``,
       leaving ty unable to find the workspace's installed packages.
    2. The directory holding this Python interpreter is *appended* to
       ``PATH``. When iommi-lsp is installed as a uv tool, the bundled
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
            print("usage: iommi-lsp graph build [path]", file=sys.stderr)
            return 2
        return _run_proxy(args.ty_command, args.workspace)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
