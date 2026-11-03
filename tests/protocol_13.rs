//! Socket framing contract for #13: bounds, timeouts and failure isolation.
//!
//! Exercises the real framing over `UnixStream::pair()` without ONNX models.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::thread;
use std::time::{Duration, Instant};

use rapidocr_worker::protocol::read_request;
use rapidocr_worker::protocol::write_response;

fn frame(meta: &[u8], image: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(meta.len() as u32).to_be_bytes());
    bytes.extend_from_slice(&(image.len() as u32).to_be_bytes());
    bytes.extend_from_slice(meta);
    bytes.extend_from_slice(image);
    bytes
}

#[test]
fn read_request_roundtrips_a_valid_frame() {
    let (mut client, mut server) = UnixStream::pair().unwrap();
    let sent = frame(b"{\"media_type\":\"image/png\"}", &[1, 2, 3, 4]);
    let writer = thread::spawn(move || {
        client.write_all(&sent).unwrap();
    });
    let (meta, image) = read_request(&mut server, 1024).unwrap();
    assert_eq!(meta, b"{\"media_type\":\"image/png\"}");
    assert_eq!(image, vec![1, 2, 3, 4]);
    writer.join().unwrap();
}

#[test]
fn read_request_rejects_oversized_declaration_without_allocating() {
    let (mut client, mut server) = UnixStream::pair().unwrap();
    let mut header = Vec::new();
    header.extend_from_slice(&16u32.to_be_bytes());
    header.extend_from_slice(&4_000_000_000u32.to_be_bytes());
    client.write_all(&header).unwrap();
    let err = read_request(&mut server, 64 * 1024 * 1024).unwrap_err();
    assert!(err.contains("frame too large"), "err: {err}");
}

#[test]
fn read_request_reports_truncated_frames() {
    let (mut client, mut server) = UnixStream::pair().unwrap();
    client.write_all(&frame(b"{}", &[1, 2, 3])[..10]).unwrap();
    drop(client);
    assert!(read_request(&mut server, 1024).is_err());
}

#[test]
fn stalled_client_hits_the_read_timeout() {
    let (_client, mut server) = UnixStream::pair().unwrap();
    server
        .set_read_timeout(Some(Duration::from_millis(100)))
        .unwrap();
    let started = Instant::now();
    let result = read_request(&mut server, 1024);
    assert!(result.is_err());
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[test]
fn write_response_writes_a_full_frame() {
    let (mut client, mut server) = UnixStream::pair().unwrap();
    let reader = thread::spawn(move || {
        let mut header = [0u8; 4];
        client.read_exact(&mut header).unwrap();
        let len = u32::from_be_bytes(header) as usize;
        let mut body = vec![0u8; len];
        client.read_exact(&mut body).unwrap();
        body
    });
    write_response(&mut server, b"{\"ok\":true}").unwrap();
    drop(server);
    assert_eq!(reader.join().unwrap(), b"{\"ok\":true}");
}
