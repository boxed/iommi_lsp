"""End-to-end echo-proxy test.

Spawns ``iommi_lsp`` as a subprocess against a tiny ``fake_ty.py`` backend,
writes LSP frames into the proxy's stdin, and reads them back from stdout.
This is the smoke test that proves the bidirectional pump works.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).parent
FAKE_TY = HERE / "fake_ty.py"


async def _read_frame(reader: asyncio.StreamReader) -> dict:
    content_length: int | None = None
    while True:
        line = await reader.readline()
        if not line:
            raise EOFError("proxy closed before sending a full frame")
        if line in (b"\r\n", b"\n"):
            break
        name, _, value = line.rstrip(b"\r\n").decode("ascii").partition(":")
        if name.strip().lower() == "content-length":
            content_length = int(value.strip())
    assert content_length is not None
    body = await reader.readexactly(content_length)
    return json.loads(body)


def _frame(payload: dict) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    return b"Content-Length: %d\r\n\r\n%s" % (len(body), body)


@pytest.mark.asyncio
async def test_echo_round_trip():
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "iommi_lsp",
        "--ty-command",
        f"{sys.executable} {FAKE_TY}",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin is not None and proc.stdout is not None
    try:
        proc.stdin.write(_frame({"jsonrpc": "2.0", "id": 1, "method": "initialize"}))
        proc.stdin.write(_frame({"jsonrpc": "2.0", "method": "ping", "params": {"n": 7}}))
        await proc.stdin.drain()

        first = await asyncio.wait_for(_read_frame(proc.stdout), timeout=5.0)
        second = await asyncio.wait_for(_read_frame(proc.stdout), timeout=5.0)

        assert first["id"] == 1
        assert first["echoed_by"] == "fake_ty", "frame must have round-tripped through the backend"
        assert second["method"] == "ping"
        assert second["params"] == {"n": 7}
        assert second["echoed_by"] == "fake_ty"

        # Tell the backend to exit; pumps should unwind cleanly.
        proc.stdin.write(_frame({"jsonrpc": "2.0", "method": "exit"}))
        await proc.stdin.drain()
        # Read the echo of the exit notification, then expect EOF.
        await _read_frame(proc.stdout)

        rc = await asyncio.wait_for(proc.wait(), timeout=5.0)
        assert rc == 0
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
