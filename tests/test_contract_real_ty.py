"""Contract test against the real ``ty server``.

Boots the proxy with the actual ty backend, opens a file in the
contract corpus, waits for diagnostics, and asserts:

* The known false positives (``Item.objects``, ``Item._meta``,
  ``item.pk``) are filtered out.
* The genuine ``Item.totally_made_up_attribute`` survives.

This is the suite that catches breakage when ``ty`` is bumped — see
DESIGN §7. Skipped automatically if ``ty`` isn't on PATH (so the rest
of the suite still runs in minimal environments).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).parent
WORKSPACE = HERE / "corpus" / "contract"
USAGE = WORKSPACE / "shop" / "usage.py"

TY_BIN = shutil.which("ty")
pytestmark = pytest.mark.skipif(
    TY_BIN is None,
    reason="real ty binary not on PATH; skipping contract test",
)


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


async def _read_until_diagnostics_for(
    reader: asyncio.StreamReader, uri: str, timeout: float = 30.0
) -> dict:
    """Drain frames until we see publishDiagnostics for *uri*."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError(f"no diagnostics for {uri} within {timeout}s")
        msg = await asyncio.wait_for(_read_frame(reader), timeout=remaining)
        if msg.get("method") != "textDocument/publishDiagnostics":
            continue
        params = msg.get("params") or {}
        if params.get("uri") == uri:
            return msg


@pytest.mark.asyncio
async def test_contract_real_ty_filters_django_false_positives():
    assert TY_BIN is not None  # for type checkers
    uri = USAGE.as_uri()

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "iommi_lsp",
        "--workspace", str(WORKSPACE),
        "--ty-command", f"{TY_BIN} server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin is not None and proc.stdout is not None

    try:
        # initialize
        proc.stdin.write(_frame({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "processId": None,
                "rootUri": WORKSPACE.as_uri(),
                "workspaceFolders": [
                    {"uri": WORKSPACE.as_uri(), "name": "contract"},
                ],
                "capabilities": {
                    "textDocument": {
                        "publishDiagnostics": {"relatedInformation": False},
                    },
                },
            },
        }))
        await proc.stdin.drain()

        init_resp = await asyncio.wait_for(_read_frame(proc.stdout), timeout=15.0)
        assert init_resp.get("id") == 1

        # initialized
        proc.stdin.write(_frame({
            "jsonrpc": "2.0", "method": "initialized", "params": {},
        }))
        await proc.stdin.drain()

        # didOpen
        source = USAGE.read_text()
        proc.stdin.write(_frame({
            "jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": uri,
                    "languageId": "python",
                    "version": 1,
                    "text": source,
                },
            },
        }))
        await proc.stdin.drain()

        msg = await _read_until_diagnostics_for(proc.stdout, uri, timeout=30.0)
        diagnostics = msg["params"]["diagnostics"]
        messages = [d.get("message", "") for d in diagnostics]

        # The genuine bug must survive.
        assert any("totally_made_up_attribute" in m for m in messages), \
            f"genuine bug must survive; got: {messages}"

        # Each false positive must be gone.
        for needle in ("`objects`", "`_meta`", "`pk`"):
            assert not any(needle in m for m in messages), \
                f"{needle} false positive should be filtered; got: {messages}"

        # Tight assertion: exactly the one real bug should remain.
        assert len(diagnostics) == 1, f"expected 1 surviving diagnostic; got: {messages}"

        # Clean shutdown.
        proc.stdin.write(_frame({"jsonrpc": "2.0", "id": 99, "method": "shutdown"}))
        await proc.stdin.drain()
        await asyncio.wait_for(_read_frame(proc.stdout), timeout=10.0)
        proc.stdin.write(_frame({"jsonrpc": "2.0", "method": "exit"}))
        await proc.stdin.drain()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            pass
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
