"""Tiny LSP-shaped echo server used as a stand-in for ``ty server`` in tests.

It reads LSP frames from stdin and echoes them back on stdout, with a
``"echoed_by":"fake_ty"`` marker added to the JSON body so the test can
verify the round trip went through the backend (rather than getting
short-circuited somewhere in the proxy). Exits when it receives a frame
whose method is ``exit``.
"""

from __future__ import annotations

import json
import sys


def _read_frame(stdin) -> bytes | None:
    content_length: int | None = None
    while True:
        line = stdin.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        name, _, value = line.rstrip(b"\r\n").decode("ascii").partition(":")
        if name.strip().lower() == "content-length":
            content_length = int(value.strip())
    if content_length is None:
        return None
    return stdin.read(content_length)


def _write_frame(stdout, body: bytes) -> None:
    stdout.write(b"Content-Length: %d\r\n\r\n" % len(body))
    stdout.write(body)
    stdout.flush()


def main() -> int:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while True:
        body = _read_frame(stdin)
        if body is None:
            return 0
        try:
            payload = json.loads(body)
        except Exception:
            payload = {"raw": body.decode("utf-8", "replace")}
        if isinstance(payload, dict):
            payload["echoed_by"] = "fake_ty"
            method = payload.get("method")
        else:
            method = None
        _write_frame(stdout, json.dumps(payload).encode("utf-8"))
        if method == "exit":
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
