"""Tests for the CompletionMatchmaker proxy hook pair."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iommi_lsp.analyzers.base import Analyzer
from iommi_lsp.interceptor import CompletionMatchmaker


def _frame(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


class _Completer(Analyzer):
    """Test analyzer that returns the items it was constructed with."""
    name = "completer"

    def __init__(self, items: list[dict]) -> None:
        self.items = items

    async def index(self, workspace_root: Path) -> None: ...
    async def on_file_changed(self, uri: str) -> None: ...
    def is_false_positive(self, uri, diag): return False

    def completions(self, uri: str, position: dict) -> list[dict]:
        return list(self.items)


@pytest.mark.asyncio
async def test_request_without_capture_passes_through():
    m = CompletionMatchmaker(analyzers=[_Completer([])])
    body = _frame({
        "jsonrpc": "2.0",
        "method": "textDocument/didOpen",
        "params": {},
    })
    out = await m.on_request(body)
    assert out is body


@pytest.mark.asyncio
async def test_completion_response_augmented_with_items():
    m = CompletionMatchmaker(
        analyzers=[_Completer([{"label": "email", "insertText": "email="}])]
    )
    # Editor → ty: completion request.
    req = _frame({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "textDocument/completion",
        "params": {
            "textDocument": {"uri": "file:///x.py"},
            "position": {"line": 0, "character": 10},
        },
    })
    await m.on_request(req)

    # ty → editor: empty completion response.
    resp = _frame({"jsonrpc": "2.0", "id": 7, "result": {"items": []}})
    out = await m.on_response(resp)
    assert out is not None
    decoded = json.loads(out)
    assert [it["label"] for it in decoded["result"]["items"]] == ["email"]


@pytest.mark.asyncio
async def test_response_merges_with_ty_items():
    m = CompletionMatchmaker(
        analyzers=[_Completer([{"label": "email"}])]
    )
    await m.on_request(_frame({
        "jsonrpc": "2.0", "id": 1, "method": "textDocument/completion",
        "params": {
            "textDocument": {"uri": "file:///x.py"},
            "position": {"line": 0, "character": 0},
        },
    }))

    resp = _frame({
        "jsonrpc": "2.0", "id": 1,
        "result": {"isIncomplete": False, "items": [{"label": "objects"}]},
    })
    decoded = json.loads(await m.on_response(resp))
    labels = [it["label"] for it in decoded["result"]["items"]]
    assert labels == ["objects", "email"]


@pytest.mark.asyncio
async def test_response_handles_list_result_shape():
    m = CompletionMatchmaker(
        analyzers=[_Completer([{"label": "email"}])]
    )
    await m.on_request(_frame({
        "jsonrpc": "2.0", "id": 2, "method": "textDocument/completion",
        "params": {
            "textDocument": {"uri": "file:///x.py"},
            "position": {"line": 0, "character": 0},
        },
    }))
    resp = _frame({"jsonrpc": "2.0", "id": 2, "result": [{"label": "obj"}]})
    decoded = json.loads(await m.on_response(resp))
    assert [it["label"] for it in decoded["result"]] == ["obj", "email"]


@pytest.mark.asyncio
async def test_response_when_ty_errored_we_substitute():
    m = CompletionMatchmaker(
        analyzers=[_Completer([{"label": "email"}])]
    )
    await m.on_request(_frame({
        "jsonrpc": "2.0", "id": 3, "method": "textDocument/completion",
        "params": {
            "textDocument": {"uri": "file:///x.py"},
            "position": {"line": 0, "character": 0},
        },
    }))
    err = _frame({
        "jsonrpc": "2.0", "id": 3,
        "error": {"code": -32601, "message": "Method not found"},
    })
    decoded = json.loads(await m.on_response(err))
    assert "error" not in decoded
    assert [it["label"] for it in decoded["result"]["items"]] == ["email"]


@pytest.mark.asyncio
async def test_response_unrelated_id_unchanged():
    m = CompletionMatchmaker(
        analyzers=[_Completer([{"label": "should-not-show"}])]
    )
    resp = _frame({"jsonrpc": "2.0", "id": 99, "result": {"items": []}})
    out = await m.on_response(resp)
    assert out is resp


@pytest.mark.asyncio
async def test_initialize_response_patched_with_completion_capability():
    m = CompletionMatchmaker(analyzers=[_Completer([{"label": "x"}])])
    await m.on_request(_frame({
        "jsonrpc": "2.0", "id": 0, "method": "initialize",
        "params": {"rootUri": "file:///tmp"},
    }))
    resp = _frame({
        "jsonrpc": "2.0", "id": 0,
        "result": {"capabilities": {"textDocumentSync": 2}},
    })
    decoded = json.loads(await m.on_response(resp))
    caps = decoded["result"]["capabilities"]
    assert "completionProvider" in caps
    assert "textDocumentSync" in caps   # didn't clobber existing capabilities


@pytest.mark.asyncio
async def test_initialize_response_keeps_existing_completion_provider():
    m = CompletionMatchmaker(analyzers=[_Completer([])])
    await m.on_request(_frame({
        "jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {},
    }))
    resp = _frame({
        "jsonrpc": "2.0", "id": 0,
        "result": {"capabilities": {
            "completionProvider": {"triggerCharacters": ["."]},
        }},
    })
    body = resp
    out = await m.on_response(body)
    # Zero-copy passthrough when ty already advertises completion.
    assert out is body


@pytest.mark.asyncio
async def test_analyzer_without_completions_method_is_fine():
    class Slim(Analyzer):
        name = "slim"
        async def index(self, workspace_root: Path) -> None: ...
        async def on_file_changed(self, uri: str) -> None: ...
        def is_false_positive(self, uri, diag): return False

    m = CompletionMatchmaker(analyzers=[Slim()])
    await m.on_request(_frame({
        "jsonrpc": "2.0", "id": 5, "method": "textDocument/completion",
        "params": {
            "textDocument": {"uri": "file:///x.py"},
            "position": {"line": 0, "character": 0},
        },
    }))
    resp = _frame({"jsonrpc": "2.0", "id": 5, "result": {"items": []}})
    out = await m.on_response(resp)
    # No completions to add → verbatim.
    assert out is resp
