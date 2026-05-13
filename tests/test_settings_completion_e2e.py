"""End-to-end completion test for Django-settings strings.

Drives the real ``iommi_lsp`` proxy with a stand-in ty backend that
returns no completions of its own. Verifies that INSTALLED_APPS items
the analyzer would produce actually reach the editor — catching
regressions where the unit tests pass but the proxy plumbing
(``didOpen`` → ``DocumentStore`` → ``CompletionMatchmaker``) fails to
wire the analyzer up.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).parent
FAKE_TY = HERE / "fake_ty_completer.py"
WORKSPACE = HERE / "corpus" / "settings_project"
SETTINGS_PY = WORKSPACE / "settings.py"


async def _read_frame(reader: asyncio.StreamReader) -> dict:
    cl: int | None = None
    while True:
        line = await reader.readline()
        if not line:
            raise EOFError("proxy closed before sending a full frame")
        if line in (b"\r\n", b"\n"):
            break
        n, _, v = line.rstrip(b"\r\n").decode("ascii").partition(":")
        if n.strip().lower() == "content-length":
            cl = int(v.strip())
    assert cl is not None
    body = await reader.readexactly(cl)
    return json.loads(body)


def _frame(payload: dict) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    return b"Content-Length: %d\r\n\r\n%s" % (len(body), body)


async def _read_until_id(
    reader: asyncio.StreamReader, msg_id: int, *, timeout: float = 10.0,
) -> dict:
    """Read frames until we see a response with the given id."""
    while True:
        frame = await asyncio.wait_for(_read_frame(reader), timeout=timeout)
        if frame.get("id") == msg_id:
            return frame


@pytest.mark.asyncio
async def test_installed_apps_completion_through_proxy():
    text = SETTINGS_PY.read_text()
    # Cursor sits at the column right after the lone ``'`` on line 1.
    cursor_line = 1
    cursor_char = len("    '")
    uri = SETTINGS_PY.as_uri()

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "iommi_lsp",
        "--workspace", str(WORKSPACE),
        "--ty-command", f"{sys.executable} {FAKE_TY}",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin is not None and proc.stdout is not None

    try:
        proc.stdin.write(_frame({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"rootUri": WORKSPACE.as_uri(), "capabilities": {}},
        }))
        proc.stdin.write(_frame({
            "jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": {"textDocument": {
                "uri": uri, "languageId": "python",
                "version": 1, "text": text,
            }},
        }))
        proc.stdin.write(_frame({
            "jsonrpc": "2.0", "id": 2, "method": "textDocument/completion",
            "params": {
                "textDocument": {"uri": uri},
                "position": {"line": cursor_line, "character": cursor_char},
            },
        }))
        await proc.stdin.drain()

        init = await _read_until_id(proc.stdout, 1)
        assert init.get("id") == 1

        completion = await _read_until_id(proc.stdout, 2)
        result = completion.get("result")
        assert isinstance(result, dict), f"expected dict result, got {result!r}"
        items = result.get("items") or []
        labels = {it["label"] for it in items}

        # Static django.contrib.* surface.
        assert "django.contrib.admin" in labels, labels
        assert "django.contrib.auth" in labels, labels
        # iommi ships an AppConfig — the analyzer always offers it.
        assert "iommi" in labels, labels
        # Workspace discovery via tests/corpus/settings_project/myapp/apps.py.
        assert "myapp" in labels, labels

        proc.stdin.write(_frame({"jsonrpc": "2.0", "method": "exit"}))
        await proc.stdin.drain()
        rc = await asyncio.wait_for(proc.wait(), timeout=5.0)
        assert rc == 0
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
