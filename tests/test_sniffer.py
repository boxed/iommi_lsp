"""Tests for the editor → ty sniffer that triggers indexing on initialize."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from iommi_lsp.interceptor import DocumentStore, EditorRequestSniffer


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
async def test_did_change_does_not_trigger_file_callback():
    # didChange happens per keystroke and the disk file hasn't moved —
    # the analyzers' reindex (Django re-scrapes + rebuilds the whole
    # workspace index, ~120 ms on a project with hundreds of models)
    # would burn that cost every keystroke for no gain. The
    # DocumentStore is enough to keep completion fresh; on_file_changed
    # is reserved for didSave (disk state actually changed).
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
    assert seen == []


@pytest.mark.asyncio
async def test_did_save_triggers_file_callback():
    seen: list[str] = []

    async def cb(uri: str) -> None:
        seen.append(uri)

    sniffer = EditorRequestSniffer(on_file_changed=cb)
    body = _frame_body({
        "jsonrpc": "2.0", "method": "textDocument/didSave",
        "params": {"textDocument": {"uri": "file:///tmp/p/foo.py"}},
    })
    await sniffer(body)
    await asyncio.sleep(0)
    assert seen == ["file:///tmp/p/foo.py"]


@pytest.mark.asyncio
async def test_document_store_tracks_open_change_close():
    store = DocumentStore()
    sniffer = EditorRequestSniffer(document_store=store)

    await sniffer(_frame_body({
        "jsonrpc": "2.0", "method": "textDocument/didOpen",
        "params": {"textDocument": {
            "uri": "file:///x.py", "languageId": "python",
            "version": 1, "text": "line one\nline two\n",
        }},
    }))
    assert store.get("file:///x.py") == "line one\nline two\n"

    # Full-document sync replaces everything.
    await sniffer(_frame_body({
        "jsonrpc": "2.0", "method": "textDocument/didChange",
        "params": {
            "textDocument": {"uri": "file:///x.py", "version": 2},
            "contentChanges": [{"text": "rewritten\n"}],
        },
    }))
    assert store.get("file:///x.py") == "rewritten\n"

    await sniffer(_frame_body({
        "jsonrpc": "2.0", "method": "textDocument/didClose",
        "params": {"textDocument": {"uri": "file:///x.py"}},
    }))
    assert store.get("file:///x.py") is None


@pytest.mark.asyncio
async def test_document_store_incremental_change():
    store = DocumentStore()
    sniffer = EditorRequestSniffer(document_store=store)

    await sniffer(_frame_body({
        "jsonrpc": "2.0", "method": "textDocument/didOpen",
        "params": {"textDocument": {
            "uri": "file:///x.py", "languageId": "python",
            "version": 1, "text": "abc\ndef\nghi\n",
        }},
    }))

    # Replace "def" on line 1 with "DEF".
    await sniffer(_frame_body({
        "jsonrpc": "2.0", "method": "textDocument/didChange",
        "params": {
            "textDocument": {"uri": "file:///x.py", "version": 2},
            "contentChanges": [{
                "range": {
                    "start": {"line": 1, "character": 0},
                    "end": {"line": 1, "character": 3},
                },
                "text": "DEF",
            }],
        },
    }))
    assert store.get("file:///x.py") == "abc\nDEF\nghi\n"


@pytest.mark.asyncio
async def test_document_store_incremental_change_sequence():
    store = DocumentStore()
    sniffer = EditorRequestSniffer(document_store=store)

    await sniffer(_frame_body({
        "jsonrpc": "2.0", "method": "textDocument/didOpen",
        "params": {"textDocument": {
            "uri": "file:///x.py", "languageId": "python",
            "version": 1, "text": "hello world",
        }},
    }))

    # Two edits applied in order: insert "X" at start, then delete the original "h".
    # Result: "Xello world".
    await sniffer(_frame_body({
        "jsonrpc": "2.0", "method": "textDocument/didChange",
        "params": {
            "textDocument": {"uri": "file:///x.py", "version": 2},
            "contentChanges": [
                {
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 0},
                    },
                    "text": "X",
                },
                {
                    "range": {
                        "start": {"line": 0, "character": 1},
                        "end": {"line": 0, "character": 2},
                    },
                    "text": "",
                },
            ],
        },
    }))
    assert store.get("file:///x.py") == "Xello world"


@pytest.mark.asyncio
async def test_unrelated_traffic_is_passed_through():
    sniffer = EditorRequestSniffer(
        on_workspace=lambda *_: (_ for _ in ()).throw(AssertionError("called")),
    )
    body = _frame_body({"jsonrpc": "2.0", "id": 1, "method": "shutdown"})
    out = await sniffer(body)
    assert out is body
