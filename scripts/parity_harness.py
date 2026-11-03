"""Parity harness: Python in-process RapidOCR vs the Rust worker (#7).

Runs CPU-only. Renders page 1 of each PDF sample at 150 DPI (images pass
through), queries both engines with the identical bytes, and writes a
digit-masked JSON report with per-sample similarity and confidence deltas.

The Python side requires the reference orchestrator package
(`ai_orchestrator`) importable; the Rust side only needs the release binary.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pymupdf

MODELS_DIR = os.environ.get("RAPIDOCR_MODELS_DIR", "/opt/models")
RUST_DEFAULTS = {
    "det": f"{MODELS_DIR}/ch_PP-OCRv4_det_infer.onnx",
    "cls": f"{MODELS_DIR}/ch_ppocr_mobile_v2.0_cls_infer.onnx",
    "rec": f"{MODELS_DIR}/ppocrv5_latin_rec_mobile.onnx",
}


def mask(text: str) -> str:
    return re.sub(r"\d", "#", text)


def sample_images(samples_dir: Path, limit: int) -> list[tuple[str, bytes]]:
    """Render page 1 of each PDF at 150 DPI; pass images through as-is."""
    items: list[tuple[str, bytes]] = []
    for path in sorted(samples_dir.iterdir()):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            with pymupdf.open(path) as doc:
                page = doc[0]
                items.append((path.name, page.get_pixmap(dpi=150).tobytes("png")))
        elif suffix in {".png", ".jpg", ".jpeg"}:
            items.append((path.name, path.read_bytes()))
        if len(items) >= limit:
            break
    return items


def query_rust(socket_path: str, image: bytes) -> dict:
    meta = json.dumps({"media_type": "image/png"}, ensure_ascii=False).encode()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(socket_path)
        client.sendall(struct.pack(">II", len(meta), len(image)) + meta + image)
        (length,) = struct.unpack(">I", client.recv(4))
        return json.loads(client.recv(length))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", default="samples")
    parser.add_argument("--limit", type=int, default=33)
    parser.add_argument(
        "--worker-bin",
        default="target/release/rapidocr-worker",
    )
    parser.add_argument("--output", default="parity_report.json")
    parser.add_argument(
        "--env-file",
        default="",
        help="Optional env file with engine settings (skipped when absent)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Intra-op threads for the Rust worker; defaults to the Python engine setting",
    )
    args = parser.parse_args()

    env_path = Path(args.env_file) if args.env_file else None
    env_lines = (
        env_path.read_text().splitlines() if env_path and env_path.is_file() else []
    )
    for line in env_lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key] = value

    threads = args.threads or int(os.environ.get("RAPIDOCR_THREADS", "8"))

    from ai_orchestrator.config import OrchestratorSettings
    from ai_orchestrator.rapidocr_client import RapidOcrClient
    from ai_orchestrator.telemetry import Telemetry

    settings = OrchestratorSettings()
    settings.rapidocr_process_pool = False
    python_client = RapidOcrClient(settings, Telemetry(settings))
    python_client.load()

    with tempfile.TemporaryDirectory() as tmp:
        rust_socket = str(Path(tmp) / "rust.sock")
        command = [
            args.worker_bin,
            "--socket",
            rust_socket,
            "--det-model",
            RUST_DEFAULTS["det"],
            "--cls-model",
            RUST_DEFAULTS["cls"],
            "--rec-model",
            RUST_DEFAULTS["rec"],
            "--threads",
            str(threads),
        ]
        env = dict(os.environ)
        if ort_dylib := env.get("ORT_DYLIB_PATH"):
            ort_dir = str(Path(ort_dylib).parent)
            existing = env.get("LD_LIBRARY_PATH")
            env["LD_LIBRARY_PATH"] = f"{existing}:{ort_dir}" if existing else ort_dir
        worker = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
        )
        for _ in range(50):
            if Path(rust_socket).exists():
                break
            time.sleep(0.2)

        report = []
        for name, image in sample_images(Path(args.samples_dir), args.limit):
            try:
                py_text, py_conf = python_client._predict(image)
            except Exception as error:  # noqa: BLE001
                py_text, py_conf = "", None
                print(f"python error {name}: {error}", flush=True)
            try:
                payload = query_rust(rust_socket, image)
                rust_text, rust_conf = (
                    payload.get("text") or "",
                    payload.get("confidence"),
                )
            except Exception as error:  # noqa: BLE001
                rust_text, rust_conf = "", None
                print(f"rust error {name}: {error}", flush=True)
            similarity = (
                difflib.SequenceMatcher(None, py_text, rust_text).ratio()
                if py_text or rust_text
                else 1.0
            )
            report.append(
                {
                    "sample": name,
                    "similarity": round(similarity, 4),
                    "python_chars": len(py_text),
                    "rust_chars": len(rust_text),
                    "confidence_delta": (
                        round(abs((py_conf or 0) - (rust_conf or 0)), 4)
                        if py_conf is not None and rust_conf is not None
                        else None
                    ),
                    "python_head": mask(py_text[:60]),
                    "rust_head": mask(rust_text[:60]),
                }
            )
            print(json.dumps(report[-1], ensure_ascii=False), flush=True)
        worker.terminate()

    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    similarities = [item["similarity"] for item in report]
    print(
        f"mean_similarity={sum(similarities) / len(similarities):.4f} samples={len(report)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
