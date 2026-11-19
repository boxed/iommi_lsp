"""Walking a workspace for the files the analyzers index.

Every analyzer that scans the workspace goes through :func:`walk_workspace`,
an ``os.walk`` that leaves out what is never the project's own source:
VCS / tool / virtualenv directories, dot-directories, and directories git
ignores.

The last one is not a nicety. A real workspace carries ignored trees —
scratch copies of the project, build output, dumps — and a walk that enters
them indexes the project several times over: on one, 6,316 of the 9,649
``.py`` files found (77 of 116 MB) were two ignored copies of the tree under
``tmp/``, and the Django index spent two thirds of its 16 s parsing them,
then listed their models and migrations beside the real ones.

Only ignored *directories* are pruned. An ignored file in a tracked directory
(a ``local_settings.py``) is still the project's and stays.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

from . import log


_log = log.get("walk")

SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "build", "dist", "site-packages",
})

# A workspace this slow to list has something wrong with it (a network
# mount, a huge untracked tree); indexing its ignored directories is the
# lesser cost.
_GIT_TIMEOUT_S = 10.0

# Indexing a workspace walks it once per analyzer scan, back to back; one
# listing serves the round. Short, so an edited ``.gitignore`` is seen by
# the next one.
_IGNORED_REUSE_S = 5.0
_ignored_recent: dict[str, tuple[float, frozenset[str]]] = {}


def ignored_dirs(root: Path) -> frozenset[str]:
    """Absolute paths of the directories under *root* that git ignores.

    Empty when *root* isn't in a git work tree, git isn't installed, or
    *root* itself is ignored — a workspace opened inside an ignored
    directory of some outer repository is still the workspace.
    """
    key = os.fspath(root)
    recent = _ignored_recent.get(key)
    now = time.monotonic()
    if recent is not None and now - recent[0] < _IGNORED_REUSE_S:
        return recent[1]
    found = _list_ignored_dirs(root)
    _ignored_recent[key] = (now, found)
    return found


def _in_git_work_tree(root: Path) -> bool:
    """Whether *root* or a parent holds a ``.git`` (a directory, or the
    file a linked work tree / submodule has). Asked before spawning git:
    outside a repository there is nothing for it to say."""
    return any((p / ".git").exists() for p in (root, *root.parents))


def _list_ignored_dirs(root: Path) -> frozenset[str]:
    if not _in_git_work_tree(root):
        return frozenset()
    try:
        proc = subprocess.run(
            [
                "git", "-C", str(root), "ls-files", "-z", "--others",
                "--ignored", "--exclude-standard", "--directory",
            ],
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as e:
        _log.debug("git ls-files in %s: %s", root, e)
        return frozenset()
    if proc.returncode != 0:
        return frozenset()
    base = os.fspath(root)
    out: set[str] = set()
    for raw in proc.stdout.split(b"\0"):
        # ``--directory`` reports a wholly ignored directory once, with a
        # trailing slash, instead of every file in it.
        if not raw.endswith(b"/"):
            continue
        rel = os.fsdecode(raw[:-1])
        if rel in ("", "."):
            return frozenset()
        out.add(os.path.join(base, rel))
    return frozenset(out)


def walk_workspace(root: Path) -> Iterator[tuple[str, list[str], list[str]]]:
    """``os.walk(root)`` without the directories that aren't the project's
    source: ``SKIP_DIRS``, dot-directories and git-ignored directories.

    Yields ``os.walk``'s ``(dirpath, dirnames, filenames)``; the caller may
    prune ``dirnames`` further, as with ``os.walk``.
    """
    ignored = ignored_dirs(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS
            and not d.startswith(".")
            and os.path.join(dirpath, d) not in ignored
        ]
        yield dirpath, dirnames, filenames
