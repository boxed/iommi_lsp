"""Fake ty backend that, after receiving ``initialize``, replies with an
``initialize`` response and then publishes a hand-crafted set of
diagnostics. Used by the end-to-end filter test.

The diagnostics are read from the env var ``DIAG_SPEW_PAYLOAD`` (a JSON
list of dicts with ``uri`` and ``diagnostics``), so the test can describe
exactly what the backend should pretend to find without us having to
hard-code anything in the script itself.
"""

from __future__ import annotations

import json
import os
import sys


def _read_frame(stdin) -> bytes | None:
    content_length: int | None = None
    while True:
        line = stdin.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        n, _, v = line.rstrip(b"\r\n").decode("ascii").partition(":")
        if n.strip().lower() == "content-length":
            content_length = int(v.strip())
    if content_length is None:
        return None
    return stdin.read(content_length)


def _write_json(stdout, payload) -> None:
    body = json.dumps(payload).encode("utf-8")
    stdout.write(b"Content-Length: %d\r\n\r\n" % len(body))
    stdout.write(body)
    stdout.flush()


def main() -> int:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    payload = json.loads(os.environ["DIAG_SPEW_PAYLOAD"])

    while True:
        body = _read_frame(stdin)
        if body is None:
            return 0
        try:
            msg = json.loads(body)
        except Exception:
            continue
        method = msg.get("method") if isinstance(msg, dict) else None

        if method == "initialize":
            _write_json(stdout, {
                "jsonrpc": "2.0",
                "id": msg.get("id"),
                "result": {
                    "capabilities": {},
                    "serverInfo": {"name": "fake-ty", "version": "0"},
                },
            })
            for spew in payload:
                _write_json(stdout, {
                    "jsonrpc": "2.0",
                    "method": "textDocument/publishDiagnostics",
                    "params": spew,
                })
        elif method == "exit":
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
