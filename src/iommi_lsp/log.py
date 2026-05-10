"""Structured logging to stderr.

LSP servers must keep stdout clean for protocol traffic, so everything
goes to stderr. We use the stdlib `logging` module configured for a
single stderr handler.
"""

from __future__ import annotations

import logging
import os
import sys


_CONFIGURED = False


def configure(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    level_name = (level or os.environ.get("IOMMI_LSP_LOG", "INFO")).upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root = logging.getLogger("iommi_lsp")
    root.setLevel(level_name)
    root.addHandler(handler)
    root.propagate = False


def get(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(f"iommi_lsp.{name}")
