"""Analyzer protocol — the contract between the proxy and a per-framework
false-positive filter (Django in v1, iommi later).

The proxy holds a list of analyzers and drops a diagnostic if **any**
analyzer flags it. This keeps the contract simple and makes it trivial
to add iommi as a second analyzer later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# A diagnostic is just the raw dict shape from ``textDocument/publishDiagnostics``.
# Using ``dict[str, Any]`` rather than the lsprotocol dataclass keeps the hot
# path zero-copy: we only validate fields we actually inspect.
Diagnostic = dict[str, Any]


@runtime_checkable
class Analyzer(Protocol):
    name: str

    async def index(self, workspace_root: Path) -> None:
        """Build/refresh the analyzer's view of the workspace."""

    async def on_file_changed(self, uri: str) -> None:
        """Update the index for a single file."""

    def is_false_positive(self, uri: str, diagnostic: Diagnostic) -> bool:
        """Return True if this diagnostic should be dropped."""
