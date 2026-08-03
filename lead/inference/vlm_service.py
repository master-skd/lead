"""Qwen-VL feature service (runs in the qwenvl env).

The closed-loop agent runs in the `lead` env, whose transformers (4.46) cannot load
Qwen3-VL. This standalone service runs in the `qwenvl` env (transformers 5.x), loads
Qwen-VL once, and serves per-frame ``vlm_hidden`` over a Unix domain socket so the
lead-env agent never has to import Qwen. See vlm_client.py for the lead-side client.

Protocol (length-prefixed raw bytes, little-endian):
    request  : uint32 H, uint32 W, then H*W*3 uint8 (RGB front-camera image)
    response : uint32 h', uint32 w', uint32 D, then h'*w'*D float16 (vlm_hidden)

Usage (qwenvl env):
    conda activate qwenvl
    python -m lead.inference.vlm_service \
        --model /path/to/Qwen3-VL-4B-Instruct \
        --socket /tmp/vlm_service.sock --prompt-mode drivable
"""

from __future__ import annotations

import argparse
import os
import socket
import struct
import sys

import numpy as np
from PIL import Image


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from the socket (or raise on EOF)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Qwen-VL model dir")
    ap.add_argument("--socket", default="/tmp/vlm_service.sock")
    ap.add_argument("--prompt-mode", choices=["drivable", "command", "3cam_drivable"], default="drivable")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from lead.inference.vlm_feature_extractor import extract_vlm_hidden, load_qwen_vl

    print(f"[vlm_service] loading Qwen-VL from {args.model} ...", flush=True)
    model, processor, image_token_id, merge = load_qwen_vl(args.model, device=args.device)
    print("[vlm_service] model loaded", flush=True)

    if os.path.exists(args.socket):
        os.remove(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    srv.listen(1)
    print(f"[vlm_service] listening on {args.socket} (prompt_mode={args.prompt_mode})", flush=True)

    while True:
        conn, _ = srv.accept()
        print("[vlm_service] client connected", flush=True)
        try:
            while True:
                header = _recv_exact(conn, 8)
                h, w = struct.unpack("<II", header)
                img_bytes = _recv_exact(conn, h * w * 3)
                img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w, 3)
                front = Image.fromarray(img, "RGB")

                feat = extract_vlm_hidden(
                    front, model, processor, image_token_id, merge,
                    prompt_mode=args.prompt_mode, device=args.device,
                )  # (h', w', D) fp16
                fh, fw, fd = feat.shape
                conn.sendall(struct.pack("<III", fh, fw, fd) + feat.tobytes())
        except (ConnectionError, BrokenPipeError):
            print("[vlm_service] client disconnected, waiting for next", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
