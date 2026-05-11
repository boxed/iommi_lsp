"""Analyzer protocol — the contract between the proxy and a per-framework
false-positive filter (Django in v1, iommi later).

The proxy holds a list of analyzers and drops a diagnostic if **any**
analyzer flags it. This keeps the contract simple and makes it trivial
to add iommi as a second analyzer later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# A diagnostic is just the raw dict shape from ``textDocument/publishDiagnostics``.
# Using ``dict[str, Any]`` rather than the lsprotocol dataclass keeps the hot
# path zero-copy: we only validate fields we actually inspect.
Diagnostic = dict[str, Any]


@dataclass
class CompletionResult:
    """Return type for ``Analyzer.completions(uri, position)``.

    *items* are LSP ``CompletionItem`` dicts. *exclusive* tells the
    matchmaker to drop any items the backend (``ty``) produced for the
    same request — used when the analyzer recognises a position where
    the backend's free-form name completions would be noise (e.g.
    inside a ``Model.objects.filter(...)`` kwarg, where every legal
    completion is a field name we already know).

    Empty + exclusive is meaningful: "we own this position; show
    nothing" rather than "we have no opinion, fall back to ty". Empty
    + not-exclusive is a no-op.
    """
    items: list[dict] = field(default_factory=list)
    exclusive: bool = False


@runtime_checkable
class Analyzer(Protocol):
    name: str

    async def index(self, workspace_root: Path) -> None:
        """Build/refresh the analyzer's view of the workspace."""

    async def on_file_changed(self, uri: str) -> None:
        """Update the index for a single file."""

    def is_false_positive(self, uri: str, diagnostic: Diagnostic) -> bool:
        """Return True if this diagnostic should be dropped."""

    def additional_diagnostics(self, uri: str) -> list[Diagnostic]:
        """Diagnostics this analyzer wants to *add* for *uri*.

        Default: none. Implement to inject framework-specific diagnostics
        (e.g. iommi refinable validation) on top of whatever the backend
        type checker found.
        """
        return []
