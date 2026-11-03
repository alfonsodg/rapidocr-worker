"""Socket contract between orchestrator clients and rapidocr-worker.

Orchestrator clients share this length-prefixed UNIX socket protocol with
the worker. This test pins the framing so both sides evolve the interface
explicitly.
"""

import struct


def encode_request(meta: bytes, image: bytes) -> bytes:
    """Request frame: >II meta_len + image_len, then meta JSON + image."""
    return struct.pack(">II", len(meta), len(image)) + meta + image


def decode_response(frame: bytes) -> tuple[int, bytes]:
    """Response frame: >I json_len + JSON {ok, text, confidence}."""
    (json_len,) = struct.unpack(">I", frame[:4])
    return json_len, frame[4 : 4 + json_len]


def test_request_framing_roundtrip() -> None:
    meta = b'{"mime":"image/png"}'
    image = bytes(range(256))
    frame = encode_request(meta, image)
    meta_len, image_len = struct.unpack(">II", frame[:8])
    assert meta_len == len(meta)
    assert image_len == len(image)
    assert frame[8 : 8 + meta_len] == meta
    assert frame[8 + meta_len :] == image


def test_response_framing_roundtrip() -> None:
    payload = b'{"ok":true,"text":"hola","confidence":0.9}'
    frame = struct.pack(">I", len(payload)) + payload
    json_len, body = decode_response(frame)
    assert json_len == len(payload)
    assert body == payload
