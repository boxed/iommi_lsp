"""Shared content-addressed AST cache used by every analyzer that
re-parses the editor buffer.

Without this, each analyzer's ``is_false_positive`` / ``additional_diagnostics``
calls ``ast.parse(source)`` independently. ty publishes one
``publishDiagnostics`` frame after analysis; each of its diagnostics
fans out through every analyzer's ``is_false_positive``. On a 10k-line
file with 50 diagnostics that's ~150 redundant parses (3 analyzers ×
50 diagnostics), pushing total analysis latency past 6 seconds.

Cache keys are ``(uri, source)`` — content-addressed, not time-based —
so an edited buffer naturally invalidates the previous tree the next
time an analyzer asks. There's no per-keystroke bookkeeping: the cache
only fills when an analyzer requests a parse, and a request with new
text replaces the old entry.

Memory bound: one tree per open uri. ``DocumentStore.did_close`` calls
:meth:`invalidate` so closed files don't linger.
"""

from __future__ import annotations

import ast


class ParseCache:
    """Content-addressed parsed-AST cache.

    Thread-safety: not designed for concurrent writes from multiple
    threads. The proxy runs analyzers on a single asyncio loop, so this
    is fine in practice.
    """

    __slots__ = ("_cache",)

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, ast.Module]] = {}

    def get(self, uri: str, source: str) -> ast.Module | None:
        """Parse *source* once per (uri, source) pair, return the tree.

        Returns ``None`` if the source has a SyntaxError — callers
        already handle that case (they're parsing user-edited buffers
        that may be mid-keystroke).
        """
        cached = self._cache.get(uri)
        if cached is not None and cached[0] is source:
            return cached[1]
        if cached is not None and cached[0] == source:
            return cached[1]
        try:
            tree = ast.parse(source)
        except SyntaxError:
            self._cache.pop(uri, None)
            return None
        self._cache[uri] = (source, tree)
        return tree

    def invalidate(self, uri: str) -> None:
        self._cache.pop(uri, None)

    def clear(self) -> None:
        self._cache.clear()
