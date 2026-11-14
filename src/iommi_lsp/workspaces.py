"""Multi-root support: one set of analyzers per workspace folder.

Every analyzer is built around a single ``workspace_root`` — its index,
its ``[tool.iommi_lsp]`` config, its iommi graph. Rather than teach each
one about several roots, the proxy keeps one *bundle* (the full analyzer
list) per workspace folder and routes every call by the file it is
about.

The routers in ``interceptor`` take a flat analyzer list and never learn
about this: each list slot is a :class:`RoutedAnalyzer` that resolves
the URI to its bundle and calls that bundle's analyzer in the same slot.

A URI under no workspace folder (a library file in ``site-packages``
outside the project, an untitled buffer) goes to the *primary* bundle —
the earliest still-open folder — which is what the single-root proxy
always did.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import log
from .analyzers.base import Analyzer
from .interceptor import _file_uri_to_path


_log = log.get("workspaces")

# Builds a fresh analyzer list for one workspace root. Every call must
# return the same classes in the same order — slot *i* of every bundle is
# what ``RoutedAnalyzer(i)`` stands for.
AnalyzerFactory = Callable[[Path], list[Analyzer]]

# Analyzer methods whose first argument is the URI they're about.
_ROUTED_METHODS = frozenset({
    "is_false_positive",
    "additional_diagnostics",
    "completions",
    "resolve_definition",
    "code_actions",
})


def workspace_venv(root: Path) -> Path | None:
    """The virtualenv inside *root* (``.venv`` or ``venv``), if any."""
    for name in (".venv", "venv"):
        candidate = root / name
        if (candidate / "bin" / "python").exists():
            return candidate
    return None


class WorkspaceSet:
    """The open workspace folders and each one's analyzer bundle."""

    def __init__(self, factory: AnalyzerFactory) -> None:
        self._factory = factory
        # Insertion-ordered: the first entry is the primary bundle.
        self._bundles: dict[Path, list[Analyzer]] = {}
        # Roots longest-first, so a nested folder wins over its parent.
        self._by_depth: list[Path] = []
        # Last lookup: one publishDiagnostics fans out to every slot for
        # every diagnostic, all for the same URI.
        self._last: tuple[str, list[Analyzer] | None] | None = None

    @property
    def roots(self) -> list[Path]:
        return list(self._bundles)

    def __contains__(self, root: Path) -> bool:
        return root in self._bundles

    async def add(self, root: Path) -> None:
        """Open *root*: build its bundle and index it. No-op if open."""
        if root in self._bundles:
            return
        bundle = self._factory(root)
        self._bundles[root] = bundle
        self._by_depth = sorted(self._bundles, key=lambda p: len(p.parts), reverse=True)
        self._last = None
        _log.info("workspace added: %s (%d open)", root, len(self._bundles))
        for a in bundle:
            # A removed-while-indexing folder stops indexing early.
            if self._bundles.get(root) is not bundle:
                return
            await a.index(root)

    def remove(self, root: Path) -> None:
        """Close *root* and let its indexes go."""
        if self._bundles.pop(root, None) is None:
            return
        self._by_depth = [p for p in self._by_depth if p != root]
        self._last = None
        _log.info("workspace removed: %s (%d open)", root, len(self._bundles))

    def root_for_path(self, path: Path) -> Path | None:
        for root in self._by_depth:
            if path.is_relative_to(root):
                return root
        return None

    def bundle_for(self, uri: str) -> list[Analyzer] | None:
        if self._last is not None and self._last[0] == uri:
            return self._last[1]
        bundle = None
        path = _file_uri_to_path(uri)
        if path is not None:
            root = self.root_for_path(path)
            if root is not None:
                bundle = self._bundles[root]
        if bundle is None:
            bundle = next(iter(self._bundles.values()), None)
        self._last = (uri, bundle)
        return bundle

    def routed(self, classes: Sequence[type]) -> list["RoutedAnalyzer"]:
        """One router per bundle slot; *classes* is the factory's class order."""
        return [RoutedAnalyzer(self, i, cls) for i, cls in enumerate(classes)]


class RoutedAnalyzer:
    """Stands in for slot *slot* of whichever bundle a URI belongs to.

    Exposes exactly the optional methods *cls* has, because the routers
    probe with ``getattr(a, "completions", None)``.
    """

    def __init__(self, workspaces: WorkspaceSet, slot: int, cls: type) -> None:
        self._workspaces = workspaces
        self._slot = slot
        self._cls = cls
        self.name = getattr(cls, "name", cls.__name__)

    def _target(self, uri: str) -> Analyzer | None:
        bundle = self._workspaces.bundle_for(uri)
        return bundle[self._slot] if bundle is not None else None

    def __getattr__(self, method: str) -> Any:
        if method not in _ROUTED_METHODS or not hasattr(self._cls, method):
            raise AttributeError(method)

        def call(uri: str, *args: Any, **kwargs: Any) -> Any:
            target = self._target(uri)
            if target is None:
                return _neutral(method)
            return getattr(target, method)(uri, *args, **kwargs)

        return call

    async def index(self, workspace_root: Path) -> None:
        await self._workspaces.add(workspace_root)

    async def on_file_changed(self, uri: str) -> None:
        target = self._target(uri)
        if target is not None:
            await target.on_file_changed(uri)


def _neutral(method: str) -> Any:
    """What a routed call answers with no workspace open at all."""
    if method == "is_false_positive":
        return False
    if method in ("additional_diagnostics", "code_actions"):
        return []
    return None


async def open_all(workspaces: WorkspaceSet, roots: Sequence[Path]) -> None:
    """Index several folders concurrently (one ``initialize``'s worth)."""
    await asyncio.gather(*(workspaces.add(r) for r in roots))
