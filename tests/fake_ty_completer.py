"""Fake ty backend that answers ``initialize`` and ``textDocument/completion``.

Used by the settings-completion end-to-end test to drive the proxy
without a real ty. Completion responses are empty — at INSTALLED_APPS
positions the SettingsAnalyzer is exclusive, so the proxy substitutes
its own items regardless of what we return here.
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
    while True:
        body = _read_frame(stdin)
        if body is None:
            return 0
        try:
            msg = json.loads(body)
        except Exception:
            continue
        if not isinstance(msg, dict):
            continue
        method = msg.get("method")
        msg_id = msg.get("id")

        if method == "initialize":
            _write_json(stdout, {
                "jsonrpc": "2.0", "id": msg_id,
                "result": {
                    "capabilities": {},
                    "serverInfo": {"name": "fake-ty-completer"},
                },
            })
        elif method == "textDocument/completion":
            _write_json(stdout, {
                "jsonrpc": "2.0", "id": msg_id,
                "result": {"isIncomplete": False, "items": []},
            })
        elif method == "exit":
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
