"""The workspace walk every analyzer scans through."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from iommi_lsp import walk
from iommi_lsp.analyzers.django import build_index
from iommi_lsp.analyzers.migrations.analyzer import discover_migrations
from iommi_lsp.walk import ignored_dirs, walk_workspace


pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

_MODELS = (
    "from django.db import models\n"
    "\n"
    "class Book(models.Model):\n"
    "    title = models.CharField(max_length=100)\n"
)


@pytest.fixture(autouse=True)
def _fresh_listing():
    """``ignored_dirs`` reuses a listing for a few seconds; a test that
    writes ``.gitignore`` and walks again must not see the last one."""
    walk._ignored_recent.clear()
    yield
    walk._ignored_recent.clear()


def _git_init(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)


def _write(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _walked(root: Path) -> set[str]:
    return {
        str(Path(dirpath, name).relative_to(root))
        for dirpath, _dirnames, filenames in walk_workspace(root)
        for name in filenames
    }


def test_git_ignored_directories_are_not_walked(tmp_path: Path):
    """A scratch copy of the project under an ignored directory is not the
    project. Walking it indexed everything twice — and, on the workspace
    this was found on, cost two thirds of the indexing time."""
    _git_init(tmp_path)
    _write(tmp_path / ".gitignore", "scratch/\n*.log\n")
    _write(tmp_path / "shop" / "models.py", _MODELS)
    _write(tmp_path / "scratch" / "before" / "shop" / "models.py", _MODELS)
    _write(tmp_path / "shop" / "debug.log")

    assert ignored_dirs(tmp_path) == frozenset({str(tmp_path / "scratch")})
    walked = _walked(tmp_path)
    assert "shop/models.py" in walked
    assert not any(name.startswith("scratch") for name in walked)
    # Only directories are pruned: an ignored file in a tracked directory
    # (a ``local_settings.py``) is still the project's.
    assert "shop/debug.log" in walked


def test_the_django_index_leaves_out_an_ignored_copy(tmp_path: Path):
    _git_init(tmp_path)
    _write(tmp_path / ".gitignore", "scratch/\n")
    _write(tmp_path / "shop" / "__init__.py")
    _write(tmp_path / "shop" / "models.py", _MODELS)
    _write(tmp_path / "scratch" / "shop" / "__init__.py")
    _write(tmp_path / "scratch" / "shop" / "models.py", _MODELS)
    _write(tmp_path / "scratch" / "shop" / "migrations" / "__init__.py")
    _write(tmp_path / "scratch" / "shop" / "migrations" / "0001_initial.py")
    _write(tmp_path / "shop" / "migrations" / "__init__.py")
    _write(tmp_path / "shop" / "migrations" / "0001_initial.py")

    models = {q for q, m in build_index(tmp_path).models.items() if not m.is_builtin}
    assert models == {"shop.models.Book"}
    assert discover_migrations(tmp_path) == {"shop": ["0001_initial"]}


def test_outside_a_repository_everything_but_the_tool_directories_is_walked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _write(tmp_path / "shop" / "models.py", _MODELS)
    _write(tmp_path / "node_modules" / "pkg" / "setup.py")
    _write(tmp_path / ".hidden" / "x.py")
    _write(tmp_path / "scratch" / "models.py", _MODELS)

    # Nothing to ask git, so it isn't run.
    def no_git(*args, **kwargs):
        raise AssertionError("git spawned outside a work tree")
    monkeypatch.setattr(walk.subprocess, "run", no_git)

    assert _walked(tmp_path) == {"shop/models.py", "scratch/models.py"}


def test_a_workspace_inside_an_ignored_directory_is_still_walked(tmp_path: Path):
    """Opening ``repo/scratch/project`` as the workspace: the outer
    repository ignores all of it, and all of it is the workspace."""
    _git_init(tmp_path)
    _write(tmp_path / ".gitignore", "scratch/\n")
    root = tmp_path / "scratch" / "project"
    _write(root / "shop" / "models.py", _MODELS)

    assert ignored_dirs(root) == frozenset()
    assert _walked(root) == {"shop/models.py"}


def test_one_listing_serves_a_round_of_walks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Indexing walks the workspace once per analyzer scan; they share one
    ``git ls-files``."""
    _git_init(tmp_path)
    _write(tmp_path / ".gitignore", "scratch/\n")
    _write(tmp_path / "scratch" / "x.py")
    calls = []
    real = subprocess.run

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)
    monkeypatch.setattr(walk.subprocess, "run", counting)

    for _ in range(3):
        assert _walked(tmp_path) == {".gitignore"}
    assert len(calls) == 1
