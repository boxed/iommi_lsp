"""``iommi_lsp graph build`` — produce the workspace's iommi graph.

Spawns a subprocess running ``python -m iommi_lsp.analyzers.iommi.reflect``
and captures its JSON output. Defaulting to ``sys.executable`` works when
iommi_lsp is installed in the same venv as iommi (the recommended setup).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ... import log
from .graph import GRAPH_FILENAME, IommiGraph, from_json, save_graph
from .reflect import DEFAULT_SEEDS


_log = log.get("iommi.build")


class GraphBuildError(RuntimeError):
    pass


def build_in_subprocess(
    *,
    python: str | None = None,
    seeds: list[str] | tuple[str, ...] = DEFAULT_SEEDS,
    timeout: float = 60.0,
    inject_iommi_lsp_path: bool = True,
) -> IommiGraph:
    """Run the reflector in a subprocess and return the resulting graph.

    When *inject_iommi_lsp_path* is true (the default, used by the
    analyzer's auto-build), the subprocess is invoked with iommi_lsp's
    install directory prepended to ``sys.path`` so it can be imported
    even when the target Python doesn't have iommi_lsp installed —
    the common case for `uv tool install`-style isolated installs of
    iommi_lsp running against the user's project venv. The target
    Python still needs iommi (and Django) on its own.
    """
    py = python or sys.executable

    if inject_iommi_lsp_path:
        import iommi_lsp as _iommi_lsp_pkg
        # `__file__` of the package = .../site-packages/iommi_lsp/__init__.py
        # so the parent of `iommi_lsp` (i.e. site-packages) is the dir we
        # need on sys.path for `import iommi_lsp` to work.
        iommi_lsp_root = str(Path(_iommi_lsp_pkg.__file__).parent.parent)
        seed_arg = ""
        if seeds and tuple(seeds) != DEFAULT_SEEDS:
            seed_arg = (
                f"\nimport sys as _s; _s.argv = ['reflect', '--seeds', "
                f"{','.join(seeds)!r}]\n"
            )
        bootstrap = (
            f"import sys; sys.path.insert(0, {iommi_lsp_root!r})"
            f"{seed_arg}\n"
            "from iommi_lsp.analyzers.iommi.reflect import main; "
            "raise SystemExit(main())"
        )
        args = [py, "-c", bootstrap]
        _log.info(
            "running graph builder (sys.path bridged): %s -c <bootstrap from %s>",
            py, iommi_lsp_root,
        )
    else:
        args = [py, "-m", "iommi_lsp.analyzers.iommi.reflect"]
        if seeds and tuple(seeds) != DEFAULT_SEEDS:
            args += ["--seeds", ",".join(seeds)]
        _log.info("running graph builder: %s", " ".join(args))

    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as e:
        raise GraphBuildError(f"could not exec {py!r}: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise GraphBuildError(f"graph build timed out after {timeout}s") from e

    if proc.returncode != 0:
        raise GraphBuildError(
            f"graph builder exited {proc.returncode}\nstderr:\n{proc.stderr}"
        )
    try:
        return from_json(proc.stdout)
    except Exception as e:
        raise GraphBuildError(f"could not parse graph output: {e}") from e


def build_for_workspace(
    workspace_root: Path,
    *,
    python: str | None = None,
    seeds: list[str] | tuple[str, ...] = DEFAULT_SEEDS,
) -> Path:
    """Build the graph and write it to ``<workspace>/.iommi_lsp-graph.json``."""
    graph = build_in_subprocess(python=python, seeds=seeds)
    out_path = workspace_root / GRAPH_FILENAME
    save_graph(graph, out_path)
    _log.info(
        "wrote iommi graph: %d classes -> %s (iommi %s)",
        len(graph.classes),
        out_path,
        graph.iommi_version,
    )
    return out_path
