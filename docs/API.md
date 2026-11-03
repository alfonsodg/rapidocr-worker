# API Documentation: rapidocr-worker Socket Protocol

`rapidocr-worker` communicates exclusively via UNIX domain sockets (`AF_UNIX`) using a length-prefixed binary framing protocol. This transport avoids HTTP overhead and Python GIL bottlenecks while maintaining sub-millisecond connection handling.

## Transport & Framing Protocol

Each request-response transaction runs sequentially over a connected stream socket.

### Request Framing

The client sends a binary frame with an 8-byte big-endian header followed by metadata and raw image bytes:

```text
+-----------------------+------------------------+-------------------+--------------------+
| meta_len (4 bytes, BE)| image_len (4 bytes, BE)| meta JSON (bytes) | image data (bytes) |
+-----------------------+------------------------+-------------------+--------------------+
```

1. **`meta_len`** (`uint32`, 4 bytes big-endian): Length in bytes of the JSON metadata string.
2. **`image_len`** (`uint32`, 4 bytes big-endian): Length in bytes of the raster image payload.
3. **`meta JSON`** (UTF-8 encoded string): Key-value options for image processing.
4. **`image data`**: Encoded image bytes (PNG, JPEG, TIFF, BMP, WebP).

#### Metadata JSON Schema

```json
{
  "media_type": "image/png"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `media_type` | `string` | No | MIME type of input image (e.g., `image/png`, `image/jpeg`). |

### Response Framing

The worker processes the image and responds with a 4-byte big-endian length prefix followed by a JSON payload:

```text
+--------------------------+-----------------------+
| payload_len (4 bytes, BE)| response JSON (bytes) |
+--------------------------+-----------------------+
```

1. **`payload_len`** (`uint32`, 4 bytes big-endian): Length in bytes of the response JSON.
2. **`response JSON`** (UTF-8 encoded string): JSON result structure.

#### Success Response

When inference succeeds, the payload contains:

```json
{
  "ok": true,
  "text": "Extracted line 1\nExtracted line 2\nExtracted line 3",
  "confidence": 0.9852,
  "elapsed_s": 0.3421,
  "det_ms": 210.4,
  "cls_ms": 12.7,
  "rec_ms": 119.0
}
```

| Field | Type | Description |
|---|---|---|
| `ok` | `boolean` | Always `true` on successful recognition. |
| `text` | `string` | Recognized text lines sorted in reading order, delimited by newlines (`\n`). |
| `confidence` | `number` | Mean recognition confidence score across all filtered text boxes (range `0.0` to `1.0`). |
| `elapsed_s` | `number` | Total processing time in seconds for detection, classification, and recognition. |
| `det_ms` | `number` | Detection stage wall-clock time in milliseconds (resize, inference and DB postprocess). |
| `cls_ms` | `number` | Direction-classification stage wall-clock time in milliseconds. |
| `rec_ms` | `number` | Recognition stage wall-clock time in milliseconds (batch preparation, inference and CTC decode). |

#### Error Response

If image decoding or model execution fails:

```json
{
  "ok": false,
  "error": "decode: image format not recognized"
}
```

| Field | Type | Description |
|---|---|---|
| `ok` | `boolean` | Always `false` when an error occurs. |
| `error` | `string` | Human-readable error description explaining the failure. |

Additional rejection paths: frames whose declared size exceeds `--max-frame-bytes` return `{"ok": false, "error": "frame too large: ..."}` before any allocation, malformed metadata returns `invalid meta json: ...`, and `--io-timeout-s` bounds stalled read/write operations so a silent client cannot block the accept loop indefinitely.

## Python Client Implementation

The reference Python client:

```python
import asyncio
import json
import struct

async def request_ocr(socket_path: str, image_bytes: bytes, media_type: str = "image/png") -> tuple[str, float]:
    reader, writer = await asyncio.open_unix_connection(socket_path)
    try:
        meta = json.dumps({"media_type": media_type}, ensure_ascii=False).encode("utf-8")
        header = struct.pack(">II", len(meta), len(image_bytes))
        writer.write(header + meta + image_bytes)
        await writer.drain()

        (length,) = struct.unpack(">I", await reader.readexactly(4))
        payload = json.loads(await reader.readexactly(length))
    finally:
        writer.close()
        await writer.wait_closed()

    if not payload.get("ok"):
        raise RuntimeError(payload.get("error") or "rapidocr worker error")
    return payload["text"], payload["confidence"]
```
