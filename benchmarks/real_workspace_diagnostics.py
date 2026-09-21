"""Boot the proxy against a real workspace and dump the surviving diagnostics.

What ty reports minus what our analyzers filter, per file — the view the
editor gets. Point it at a handful of real files and read the output for
false positives.

Usage::

    python benchmarks/real_workspace_diagnostics.py FILE [FILE ...]
    python benchmarks/real_workspace_diagnostics.py --workspace PATH FILE ...

The backing type checker defaults to the ``ty`` next to the installed
``iommi_lsp`` tool, which is where a ty-semantic fork build lives.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKSPACE = Path.home() / "Projects/dryft"
TY = Path.home() / ".local/share/uv/tools/iommi-lsp/bin/ty"


def _frame(payload: dict) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    return b"Content-Length: %d\r\n\r\n%s" % (len(body), body)


async def _read_frame(reader: asyncio.StreamReader) -> dict:
    cl = None
    while True:
        line = await reader.readline()
        if not line:
            raise EOFError("proxy closed")
        if line in (b"\r\n", b"\n"):
            break
        n, _, v = line.rstrip(b"\r\n").decode("ascii").partition(":")
        if n.strip().lower() == "content-length":
            cl = int(v.strip())
    assert cl is not None
    return json.loads(await reader.readexactly(cl))


async def main(paths: list[Path]) -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "iommi_lsp",
        "--workspace", str(WORKSPACE),
        "--ty-command", f"{TY} server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=str(WORKSPACE),
    )
    assert proc.stdin and proc.stdout

    proc.stdin.write(_frame({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "processId": None,
            "rootUri": WORKSPACE.as_uri(),
            "workspaceFolders": [{"uri": WORKSPACE.as_uri(), "name": "dryft"}],
            "capabilities": {
                "textDocument": {"publishDiagnostics": {"relatedInformation": False}},
            },
        },
    }))
    await proc.stdin.drain()
    await asyncio.wait_for(_read_frame(proc.stdout), timeout=120)
    proc.stdin.write(_frame({"jsonrpc": "2.0", "method": "initialized", "params": {}}))
    await proc.stdin.drain()

    uris = {}
    for i, p in enumerate(paths):
        uri = p.as_uri()
        uris[uri] = p
        proc.stdin.write(_frame({
            "jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": {"textDocument": {
                "uri": uri, "languageId": "python", "version": 1,
                "text": p.read_text(encoding="utf-8"),
            }},
        }))
    await proc.stdin.drain()

    seen: dict[str, list] = {}
    deadline = asyncio.get_running_loop().time() + 600
    while len(seen) < len(uris):
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        try:
            msg = await asyncio.wait_for(_read_frame(proc.stdout), timeout=remaining)
        except asyncio.TimeoutError:
            break
        if msg.get("method") != "textDocument/publishDiagnostics":
            continue
        params = msg.get("params") or {}
        uri = params.get("uri")
        if uri in uris:
            seen[uri] = params.get("diagnostics") or []

    for uri, diags in seen.items():
        path = uris[uri]
        try:
            rel = path.relative_to(WORKSPACE)
        except ValueError:
            rel = path
        lines = path.read_text(encoding="utf-8").splitlines()
        print(f"\n===== {rel}  ({len(diags)} diagnostics)")
        for d in sorted(diags, key=lambda d: d["range"]["start"]["line"]):
            ln = d["range"]["start"]["line"]
            src = lines[ln].strip() if ln < len(lines) else ""
            print(f"  {rel}:{ln+1}  [{d.get('code')}|{d.get('source')}|sev={d.get('severity')}|tags={d.get('tags')}] {d.get('message','').splitlines()[0]}")
            print(f"      | {src}")
    missing = set(uris) - set(seen)
    for uri in missing:
        print(f"\n===== {uris[uri]}  NO DIAGNOSTICS RECEIVED")

    proc.stdin.write(_frame({"jsonrpc": "2.0", "id": 99, "method": "shutdown", "params": {}}))
    await proc.stdin.drain()
    try:
        proc.terminate()
    except ProcessLookupError:
        pass
    await proc.wait()
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--workspace"]:
        WORKSPACE = Path(args[1]).resolve()
        args = args[2:]
    sys.exit(asyncio.run(main([Path(a).resolve() for a in args])))
