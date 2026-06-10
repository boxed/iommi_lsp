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


# --- pull diagnostics (LSP 3.17: textDocument/diagnostic) -------------------
#
# A pull-capable client (e.g. one advertising ``textDocument.diagnostic``)
# makes ty answer via request/response instead of pushing
# ``publishDiagnostics`` — ty even stops pushing once it registers the pull
# provider. The response carries no URI, so the interceptor pairs it with the
# originating request seen on the editor→ty side.


def _pull_request(msg_id, uri: str = "file:///x.py") -> bytes:
    return _frame_body({
        "jsonrpc": "2.0",
        "id": msg_id,
        "method": "textDocument/diagnostic",
        "params": {"textDocument": {"uri": uri}},
    })


def _pull_response(msg_id, diagnostics: list) -> bytes:
    return _frame_body({
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {"kind": "full", "resultId": "r1", "items": diagnostics},
    })


@pytest.mark.asyncio
async def test_pull_request_is_forwarded_verbatim():
    interceptor = DiagnosticInterceptor(analyzers=[_Drop()])
    body = _pull_request(1)
    out = await interceptor.on_request(body)
    assert out is body


@pytest.mark.asyncio
async def test_pull_response_drops_only_flagged():
    interceptor = DiagnosticInterceptor(analyzers=[_Drop()])
    await interceptor.on_request(_pull_request(7))
    out = await interceptor(_pull_response(7, [
        _diag("a", code="drop:noise"),
        _diag("b", code="real-bug"),
        _diag("c", code="drop:also-noise"),
    ]))
    assert out is not None
    decoded = json.loads(out)
    msgs = [d["message"] for d in decoded["result"]["items"]]
    assert msgs == ["b"]


@pytest.mark.asyncio
async def test_pull_response_without_matching_request_passes_through():
    """No editor→ty request was seen for this id, so we have no URI to filter
    against — forward untouched rather than guess."""
    interceptor = DiagnosticInterceptor(analyzers=[_Drop()])
    body = _pull_response(99, [_diag("a", code="drop:noise")])
    out = await interceptor(body)
    assert out is body


@pytest.mark.asyncio
async def test_pull_unchanged_report_passes_through():
    interceptor = DiagnosticInterceptor(analyzers=[_Drop()])
    await interceptor.on_request(_pull_request(3))
    body = _frame_body({
        "jsonrpc": "2.0",
        "id": 3,
        "result": {"kind": "unchanged", "resultId": "r1"},
    })
    out = await interceptor(body)
    assert out is body


@pytest.mark.asyncio
async def test_pull_response_filters_related_documents():
    interceptor = DiagnosticInterceptor(analyzers=[_Drop()])
    await interceptor.on_request(_pull_request(5, uri="file:///main.py"))
    body = _frame_body({
        "jsonrpc": "2.0",
        "id": 5,
        "result": {
            "kind": "full",
            "items": [_diag("keep-main", code="real"), _diag("x", code="drop:1")],
            "relatedDocuments": {
                "file:///other.py": {
                    "kind": "full",
                    "items": [_diag("y", code="drop:2"), _diag("keep-other", code="real")],
                },
            },
        },
    })
    out = await interceptor(body)
    decoded = json.loads(out)
    assert [d["message"] for d in decoded["result"]["items"]] == ["keep-main"]
    related = decoded["result"]["relatedDocuments"]["file:///other.py"]
    assert [d["message"] for d in related["items"]] == ["keep-other"]


@pytest.mark.asyncio
async def test_workspace_pull_response_filters_per_document():
    interceptor = DiagnosticInterceptor(analyzers=[_Drop()])
    await interceptor.on_request(_frame_body({
        "jsonrpc": "2.0",
        "id": 11,
        "method": "workspace/diagnostic",
        "params": {"previousResultIds": []},
    }))
    body = _frame_body({
        "jsonrpc": "2.0",
        "id": 11,
        "result": {
            "items": [
                {"kind": "full", "uri": "file:///a.py", "version": 1,
                 "items": [_diag("keep-a", code="real"), _diag("x", code="drop:1")]},
                {"kind": "full", "uri": "file:///b.py", "version": 1,
                 "items": [_diag("y", code="drop:2")]},
            ],
        },
    })
    out = await interceptor(body)
    decoded = json.loads(out)
    reports = decoded["result"]["items"]
    assert [d["message"] for d in reports[0]["items"]] == ["keep-a"]
    assert reports[1]["items"] == []


@pytest.mark.asyncio
async def test_completion_response_with_id_is_not_treated_as_pull():
    """A response whose id we never tracked as a pull request must pass
    through — we must not accidentally rewrite completion/hover results."""
    interceptor = DiagnosticInterceptor(analyzers=[_Drop()])
    body = _frame_body({"jsonrpc": "2.0", "id": 42, "result": {"items": ["x"]}})
    out = await interceptor(body)
    assert out is body
