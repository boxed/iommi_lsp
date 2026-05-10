"""LSP base-protocol framing: ``Content-Length: N\\r\\n\\r\\n<body>``.

Parses just enough of the headers to find the body length. Other headers
(``Content-Type``) are tolerated and ignored. Bodies are returned as raw
bytes so the caller decides whether to ``json.loads`` them — for the
proxy hot path we forward most messages without parsing.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any


class FramingError(Exception):
    """The peer sent something that is not a valid LSP frame."""


async def read_message(reader: asyncio.StreamReader) -> bytes | None:
    """Read one LSP frame and return its body bytes.

    Returns ``None`` on a clean EOF (peer closed before any header).
    Raises :class:`FramingError` on a truncated or malformed frame.
    """
    content_length: int | None = None

    # Headers: lines ending in CRLF, terminated by an empty line.
    while True:
        line = await reader.readline()
        if not line:
            if content_length is None:
                return None
            raise FramingError("EOF inside headers")
        if line in (b"\r\n", b"\n"):
            break
        # Tolerate either CRLF or LF, even though the spec says CRLF.
        line = line.rstrip(b"\r\n")
        if not line:
            break
        try:
            name, _, value = line.decode("ascii").partition(":")
        except UnicodeDecodeError as e:
            raise FramingError(f"non-ASCII header: {line!r}") from e
        if name.strip().lower() == "content-length":
            try:
                content_length = int(value.strip())
            except ValueError as e:
                raise FramingError(f"bad Content-Length: {value!r}") from e

    if content_length is None:
        raise FramingError("missing Content-Length header")
    if content_length < 0:
        raise FramingError(f"negative Content-Length: {content_length}")

    body = await reader.readexactly(content_length)
    return body


def encode_message(body: bytes) -> bytes:
    """Wrap a JSON body in the LSP base-protocol frame."""
    return b"Content-Length: %d\r\n\r\n%s" % (len(body), body)


def encode_json(payload: Any) -> bytes:
    """Convenience: ``encode_message(json.dumps(payload).encode())``."""
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return encode_message(body)


async def write_message(writer: asyncio.StreamWriter, body: bytes) -> None:
    writer.write(encode_message(body))
    await writer.drain()
