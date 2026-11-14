"""Fake ty backend that asks for per-workspace configuration, like ty does.

On ``initialize`` it replies, then sends a ``workspace/configuration``
request with one ``section: "ty"`` item per workspace folder. When the
answer comes back it publishes it, JSON-encoded, as the message of a
single diagnostic on ``file:///__config_report__`` — so a test reading
the proxy's output sees exactly what reached the backend.
"""

from __future__ import annotations

import json
import sys

from diagnostic_spewer import _read_frame, _write_json


REPORT_URI = "file:///__config_report__"


def main() -> int:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while True:
        body = _read_frame(stdin)
        if body is None:
            return 0
        msg = json.loads(body)
        method = msg.get("method")
        if method == "initialize":
            folders = msg["params"].get("workspaceFolders") or []
            _write_json(stdout, {
                "jsonrpc": "2.0", "id": msg["id"],
                "result": {"capabilities": {}, "serverInfo": {"name": "fake-ty"}},
            })
            _write_json(stdout, {
                "jsonrpc": "2.0", "id": "cfg-1", "method": "workspace/configuration",
                "params": {"items": [
                    {"scopeUri": f["uri"], "section": "ty"} for f in folders
                ]},
            })
        elif method is None and msg.get("id") == "cfg-1":
            _write_json(stdout, {
                "jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
                "params": {"uri": REPORT_URI, "diagnostics": [{
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 1},
                    },
                    "severity": 3,
                    "code": "fake-config-report",
                    "message": json.dumps(msg.get("result")),
                    "source": "fake-ty",
                }]},
            })
        elif method == "exit":
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
