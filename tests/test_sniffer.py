"""Tests for the editor → ty sniffer that triggers indexing on initialize."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from iommi_lsp.interceptor import EditorRequestSniffer


def _frame_body(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


@pytest.mark.asyncio
async def test_initialize_triggers_workspace_callback():
    seen: list[Path] = []

    async def cb(root: Path) -> None:
        seen.append(root)

    sniffer = EditorRequestSniffer(on_workspace=cb)

    body = _frame_body(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"rootUri": "file:///tmp/myproj", "capabilities": {}},
        }
    )
    out = await sniffer(body)
    # Forwarded verbatim.
    assert out is body

    # Background task — give it a tick to land.
    await asyncio.sleep(0)
    assert seen == [Path("/tmp/myproj")]


@pytest.mark.asyncio
async def test_workspace_folders_preferred_over_root_uri():
    seen: list[Path] = []

    async def cb(root: Path) -> None:
        seen.append(root)

    sniffer = EditorRequestSniffer(on_workspace=cb)

    body = _frame_body(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "rootUri": "file:///tmp/old",
                "workspaceFolders": [
                    {"uri": "file:///tmp/new", "name": "new"},
                ],
            },
        }
    )
    await sniffer(body)
    await asyncio.sleep(0)
    assert seen == [Path("/tmp/new")]


@pytest.mark.asyncio
async def test_initialize_only_triggers_once():
    n = 0

    async def cb(root: Path) -> None:
        nonlocal n
        n += 1

    sniffer = EditorRequestSniffer(on_workspace=cb)
    body = _frame_body({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"rootUri": "file:///tmp/p"},
    })
    await sniffer(body)
    await sniffer(body)
    await asyncio.sleep(0)
    assert n == 1


@pytest.mark.asyncio
async def test_did_change_triggers_file_callback():
    seen: list[str] = []

    async def cb(uri: str) -> None:
        seen.append(uri)

    sniffer = EditorRequestSniffer(on_file_changed=cb)
    body = _frame_body({
        "jsonrpc": "2.0", "method": "textDocument/didChange",
        "params": {
            "textDocument": {"uri": "file:///tmp/p/foo.py", "version": 2},
            "contentChanges": [],
        },
    })
    await sniffer(body)
    await asyncio.sleep(0)
    assert seen == ["file:///tmp/p/foo.py"]


@pytest.mark.asyncio
async def test_unrelated_traffic_is_passed_through():
    sniffer = EditorRequestSniffer(
        on_workspace=lambda *_: (_ for _ in ()).throw(AssertionError("called")),
    )
    body = _frame_body({"jsonrpc": "2.0", "id": 1, "method": "shutdown"})
    out = await sniffer(body)
    assert out is body
