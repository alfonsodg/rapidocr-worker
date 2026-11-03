//! Socket framing, input bounds and metadata validation for the worker protocol (#13).

use std::io::{Read, Write};

/// Read one request frame: `>II` header, meta JSON, image bytes.
///
/// The declared frame size is validated against `max_frame_bytes` before any
/// buffer allocation, so a corrupt or hostile header cannot force a huge
/// allocation.
pub fn read_request(
    stream: &mut impl Read,
    max_frame_bytes: usize,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let mut header = [0u8; 8];
    stream
        .read_exact(&mut header)
        .map_err(|e| format!("read header: {e}"))?;
    let meta_len = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as u64;
    let image_len = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as u64;
    let total = meta_len + image_len;
    if total > max_frame_bytes as u64 {
        return Err(format!(
            "frame too large: {total} bytes exceeds max_frame_bytes={max_frame_bytes}"
        ));
    }
    let mut meta = vec![0u8; meta_len as usize];
    stream
        .read_exact(&mut meta)
        .map_err(|e| format!("read meta: {e}"))?;
    let mut image = vec![0u8; image_len as usize];
    stream
        .read_exact(&mut image)
        .map_err(|e| format!("read image: {e}"))?;
    Ok((meta, image))
}

/// Validate request metadata: empty is allowed, otherwise it must be UTF-8 JSON.
pub fn validate_meta(meta: &[u8]) -> Result<(), String> {
    if meta.is_empty() {
        return Ok(());
    }
    serde_json::from_slice::<serde_json::Value>(meta)
        .map(|_| ())
        .map_err(|e| format!("invalid meta json: {e}"))
}

/// Encode a length-prefixed response frame (`>I` payload length + payload).
pub fn encode_response(payload: &[u8]) -> Vec<u8> {
    let mut response = Vec::with_capacity(4 + payload.len());
    response.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    response.extend_from_slice(payload);
    response
}

/// Write a length-prefixed response frame to the stream.
pub fn write_response(stream: &mut impl Write, payload: &[u8]) -> Result<(), String> {
    stream
        .write_all(&encode_response(payload))
        .map_err(|e| format!("write response: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_meta_accepts_empty_and_json() {
        assert!(validate_meta(b"").is_ok());
        assert!(validate_meta(b"{\"media_type\":\"image/png\"}").is_ok());
        assert!(validate_meta(b"not json").is_err());
    }

    #[test]
    fn encode_response_prefixes_the_payload_length() {
        let out = encode_response(b"{\"ok\":true}");
        assert_eq!(&out[..4], &(11u32).to_be_bytes());
        assert_eq!(&out[4..], b"{\"ok\":true}");
    }
}
