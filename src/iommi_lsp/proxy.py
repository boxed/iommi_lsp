"""Bidirectional LSP proxy.

The editor talks to us over stdio; we spawn ``ty server`` (or whatever
the user configured) as a subprocess and shuttle frames in both
directions. v1 is a pure echo proxy — interceptors get wired in at
milestone 2.
"""

from __future__ import annotations

import asyncio
import shlex
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence

from . import jsonrpc, log


_log = log.get("proxy")

# Hook signature: takes the raw JSON body bytes flowing in one direction
# and returns the bytes that should actually be forwarded, or ``None`` to
# drop the message entirely. Returning the input unchanged is the
# zero-cost default.
Hook = Callable[[bytes], Awaitable[bytes | None]]


async def _passthrough(body: bytes) -> bytes | None:
    return body


async def _pump(
    name: str,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    hook: Hook,
) -> None:
    """Move frames from ``reader`` to ``writer`` until EOF."""
    try:
        while True:
            body = await jsonrpc.read_message(reader)
            if body is None:
                _log.debug("%s: EOF", name)
                return
            try:
                forwarded = await hook(body)
            except Exception:
                _log.exception("%s: hook crashed; forwarding original", name)
                forwarded = body
            if forwarded is None:
                continue
            await jsonrpc.write_message(writer, forwarded)
    except (asyncio.IncompleteReadError, jsonrpc.FramingError) as e:
        _log.warning("%s: framing terminated: %s", name, e)
    except ConnectionResetError:
        _log.info("%s: connection reset", name)
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _stdio_streams() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Wrap process stdin/stdout as asyncio streams."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer
    )
    transport, protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout.buffer
    )
    writer = asyncio.StreamWriter(transport, protocol, None, loop)
    return reader, writer


async def run_with_streams(
    ty_command: Sequence[str],
    editor_reader: asyncio.StreamReader,
    editor_writer: asyncio.StreamWriter,
    *,
    editor_to_ty_hook: Hook = _passthrough,
    ty_to_editor_hook: Hook = _passthrough,
    env: Mapping[str, str] | None = None,
    on_writer_ready: Callable[[asyncio.StreamWriter], None] | None = None,
) -> int:
    """Run the proxy against pre-built editor streams. Returns ty's exit code.

    Split from :func:`run` so tests can inject in-memory streams instead
    of going through stdio.

    *on_writer_ready* — optional hook called with the live editor writer
    once it's set up but before the message pumps start. Used by the
    matchmaker to inject synthetic responses for known-exclusive
    completion positions without round-tripping through ty (the dominant
    cost on first-popup latency in large settings.py files).
    """
    _log.info("spawning backend: %s", " ".join(shlex.quote(p) for p in ty_command))
    proc = await asyncio.create_subprocess_exec(
        *ty_command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=None,  # let ty's stderr passthrough — useful for debugging
        env=env,
    )
    assert proc.stdin is not None and proc.stdout is not None

    if on_writer_ready is not None:
        on_writer_ready(editor_writer)

    pump_in = asyncio.create_task(
        _pump("editor→ty", editor_reader, proc.stdin, editor_to_ty_hook),
        name="pump-editor-to-ty",
    )
    pump_out = asyncio.create_task(
        _pump("ty→editor", proc.stdout, editor_writer, ty_to_editor_hook),
        name="pump-ty-to-editor",
    )

    # Whichever pump finishes first ends the session — once one direction
    # is dead, the protocol is unrecoverable.
    done, pending = await asyncio.wait(
        {pump_in, pump_out}, return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    for task in pending:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            _log.exception("pump cleanup raised")

    try:
        rc = await asyncio.wait_for(proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        _log.warning("backend did not exit within 5s; killing")
        proc.kill()
        rc = await proc.wait()
    _log.info("backend exited with code %s", rc)
    return rc


async def run(
    ty_command: Sequence[str],
    *,
    editor_to_ty_hook: Hook = _passthrough,
    ty_to_editor_hook: Hook = _passthrough,
    env: Mapping[str, str] | None = None,
    on_writer_ready: Callable[[asyncio.StreamWriter], None] | None = None,
) -> int:
    """Run the proxy on the process's stdio."""
    editor_reader, editor_writer = await _stdio_streams()
    return await run_with_streams(
        ty_command,
        editor_reader,
        editor_writer,
        editor_to_ty_hook=editor_to_ty_hook,
        ty_to_editor_hook=ty_to_editor_hook,
        env=env,
        on_writer_ready=on_writer_ready,
    )
