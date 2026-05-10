import asyncio
import json

import pytest

from iommi_lsp import jsonrpc


def _stream_with(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


async def test_round_trip():
    payload = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    encoded = jsonrpc.encode_json(payload)
    body = await jsonrpc.read_message(_stream_with(encoded))
    assert body is not None
    assert json.loads(body) == payload


async def test_two_messages_back_to_back():
    a = jsonrpc.encode_json({"a": 1})
    b = jsonrpc.encode_json({"b": 2})
    reader = _stream_with(a + b)

    first = await jsonrpc.read_message(reader)
    second = await jsonrpc.read_message(reader)
    third = await jsonrpc.read_message(reader)

    assert json.loads(first) == {"a": 1}
    assert json.loads(second) == {"b": 2}
    assert third is None  # clean EOF


async def test_clean_eof_returns_none():
    body = await jsonrpc.read_message(_stream_with(b""))
    assert body is None


async def test_extra_headers_are_tolerated():
    body = b'{"x":1}'
    raw = (
        b"Content-Length: 7\r\n"
        b"Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n"
        b"\r\n" + body
    )
    got = await jsonrpc.read_message(_stream_with(raw))
    assert got == body


async def test_missing_content_length_raises():
    raw = b"X-Other: 1\r\n\r\nhello"
    with pytest.raises(jsonrpc.FramingError):
        await jsonrpc.read_message(_stream_with(raw))


async def test_truncated_body_raises():
    raw = b"Content-Length: 10\r\n\r\nshort"
    with pytest.raises((jsonrpc.FramingError, asyncio.IncompleteReadError)):
        await jsonrpc.read_message(_stream_with(raw))


async def test_unicode_body_round_trips():
    payload = {"msg": "héllo — 世界"}
    encoded = jsonrpc.encode_json(payload)
    body = await jsonrpc.read_message(_stream_with(encoded))
    assert body is not None
    assert json.loads(body) == payload
