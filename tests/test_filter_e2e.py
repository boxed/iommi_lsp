"""End-to-end test of the full filter pipeline.

Drives the real ``iommi-lsp`` proxy with a stand-in backend that
publishes synthetic diagnostics. Verifies that magic-attribute false
positives are removed while genuine diagnostics survive.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).parent
SPEWER = HERE / "diagnostic_spewer.py"
WORKSPACE = HERE / "corpus" / "basic_django"
USAGE_PY = WORKSPACE / "usage.py"


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


def _diag_at(text: str, line: int, needle: str, *, code: str = "unresolved-attribute") -> dict:
    src_line = text.splitlines()[line]
    start = src_line.index(needle)
    return {
        "code": code,
        "message": f"unresolved attribute {needle!r}",
        "range": {
            "start": {"line": line, "character": start},
            "end": {"line": line, "character": start + len(needle)},
        },
        "severity": 1,
        "source": "ty",
    }


@pytest.mark.asyncio
async def test_filter_drops_magic_attrs_keeps_real_bugs():
    src = USAGE_PY.read_text()
    uri = USAGE_PY.as_uri()

    # Magic attr — should be filtered.
    objects_diag = _diag_at(src, line=4, needle="objects")
    # Real bug — `bogus_typo` is not a real attribute.
    typo_diag = _diag_at(src, line=10, needle="bogus_typo")

    spew = [{
        "uri": uri,
        "diagnostics": [objects_diag, typo_diag],
    }]
    env = {**os.environ, "DIAG_SPEW_PAYLOAD": json.dumps(spew)}

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "iommi_lsp",
        "--workspace", str(WORKSPACE),
        "--ty-command", f"{sys.executable} {SPEWER}",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    assert proc.stdin is not None and proc.stdout is not None

    try:
        proc.stdin.write(_frame({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "rootUri": WORKSPACE.as_uri(),
                "capabilities": {},
            },
        }))
        await proc.stdin.drain()

        # First: the initialize response.
        init_resp = await asyncio.wait_for(_read_frame(proc.stdout), timeout=10.0)
        assert init_resp.get("id") == 1

        # Then: the (filtered) publishDiagnostics.
        diag_msg = await asyncio.wait_for(_read_frame(proc.stdout), timeout=10.0)
        assert diag_msg["method"] == "textDocument/publishDiagnostics"
        kept = diag_msg["params"]["diagnostics"]
        kept_messages = [d["message"] for d in kept]

        assert any("bogus_typo" in m for m in kept_messages), \
            f"genuine bug must survive; got: {kept_messages}"
        assert not any("'objects'" in m for m in kept_messages), \
            f"User.objects must be filtered; got: {kept_messages}"
        assert len(kept) == 1

        # Clean shutdown.
        proc.stdin.write(_frame({"jsonrpc": "2.0", "method": "exit"}))
        await proc.stdin.drain()
        rc = await asyncio.wait_for(proc.wait(), timeout=5.0)
        assert rc == 0
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
