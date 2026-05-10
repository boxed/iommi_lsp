"""Unit tests for DiagnosticInterceptor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iommi_lsp.analyzers.base import Analyzer, Diagnostic
from iommi_lsp.interceptor import DiagnosticInterceptor


def _frame_body(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _diag(message: str, code: str = "unresolved-attribute") -> Diagnostic:
    return {
        "range": {
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 1},
        },
        "severity": 1,
        "code": code,
        "message": message,
        "source": "ty",
    }


class _Drop(Analyzer):
    """Test double — drops diagnostics whose code starts with ``drop:``."""

    name = "drop"

    async def index(self, workspace_root: Path) -> None: ...
    async def on_file_changed(self, uri: str) -> None: ...

    def is_false_positive(self, uri: str, diagnostic: Diagnostic) -> bool:
        return str(diagnostic.get("code", "")).startswith("drop:")


@pytest.mark.asyncio
async def test_non_diagnostic_message_is_passed_through_verbatim():
    interceptor = DiagnosticInterceptor()
    body = _frame_body({"jsonrpc": "2.0", "id": 1, "result": {"hover": "x"}})
    out = await interceptor(body)
    # Identity, not equality: no re-serialization on the hot path.
    assert out is body


@pytest.mark.asyncio
async def test_diagnostics_with_no_analyzers_pass_through_verbatim():
    interceptor = DiagnosticInterceptor()
    body = _frame_body(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {"uri": "file:///x.py", "diagnostics": [_diag("noisy")]},
        }
    )
    out = await interceptor(body)
    assert out is body


@pytest.mark.asyncio
async def test_diagnostics_drop_only_flagged():
    interceptor = DiagnosticInterceptor(analyzers=[_Drop()])
    payload = {
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {
            "uri": "file:///x.py",
            "diagnostics": [
                _diag("a", code="drop:noise"),
                _diag("b", code="real-bug"),
                _diag("c", code="drop:also-noise"),
            ],
        },
    }
    out = await interceptor(_frame_body(payload))
    assert out is not None
    decoded = json.loads(out)
    msgs = [d["message"] for d in decoded["params"]["diagnostics"]]
    assert msgs == ["b"]


@pytest.mark.asyncio
async def test_drops_all_yields_empty_list_not_missing_field():
    interceptor = DiagnosticInterceptor(analyzers=[_Drop()])
    payload = {
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {
            "uri": "file:///x.py",
            "diagnostics": [_diag("a", code="drop:1"), _diag("b", code="drop:2")],
        },
    }
    out = await interceptor(_frame_body(payload))
    decoded = json.loads(out)
    assert decoded["params"]["diagnostics"] == []


@pytest.mark.asyncio
async def test_invalid_json_is_passed_through():
    interceptor = DiagnosticInterceptor(analyzers=[_Drop()])
    out = await interceptor(b"{not json")
    assert out == b"{not json"


@pytest.mark.asyncio
async def test_missing_diagnostics_field_is_safe():
    interceptor = DiagnosticInterceptor(analyzers=[_Drop()])
    body = _frame_body(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {"uri": "file:///x.py"},
        }
    )
    out = await interceptor(body)
    # No diagnostics in, no diagnostics out — verbatim forwarding.
    assert out is body
