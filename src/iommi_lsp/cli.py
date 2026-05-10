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
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__, log, proxy
from .analyzers.django import DjangoAnalyzer, build_index
from .interceptor import DiagnosticInterceptor, EditorRequestSniffer


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
    return p


def _run_proxy(ty_command_str: str, workspace: Path | None) -> int:
    ty_command = shlex.split(ty_command_str)
    if not ty_command:
        print("error: --ty-command must not be empty", file=sys.stderr)
        return 2

    analyzer = DjangoAnalyzer(workspace_root=workspace or Path.cwd())
    interceptor = DiagnosticInterceptor(analyzers=[analyzer])
    sniffer = EditorRequestSniffer(
        on_workspace=analyzer.index,
        on_file_changed=analyzer.on_file_changed,
    )

    if workspace is not None:
        # Eager build for explicit --workspace; the sniffer still listens
        # for editor-side workspace changes but won't override.
        asyncio.run(_eager_index_then_serve(
            ty_command, analyzer, workspace, interceptor, sniffer
        ))
        return 0
    return asyncio.run(proxy.run(
        ty_command,
        editor_to_ty_hook=sniffer,
        ty_to_editor_hook=interceptor,
    ))


async def _eager_index_then_serve(
    ty_command, analyzer, workspace, interceptor, sniffer
) -> int:
    await analyzer.index(workspace)
    return await proxy.run(
        ty_command,
        editor_to_ty_hook=sniffer,
        ty_to_editor_hook=interceptor,
    )


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
        return _run_proxy(args.ty_command, args.workspace)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
