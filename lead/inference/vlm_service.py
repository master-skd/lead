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
    from lead.inference.vlm_feature_extractor import (
        check_p6_batch_input_parity,
        extract_vlm_hidden,
        extract_vlm_hidden_batch,
        load_qwen_vl,
        prepare_p6_batch_inputs,
        prepare_p6_batch_prompt,
    )

    print(f"[vlm_service] loading Qwen-VL from {args.model} ...", flush=True)
    model, processor, image_token_id, merge = load_qwen_vl(args.model, device=args.device)
    from lead.common.stage_timing import StageTimingRecorder

    recorder = StageTimingRecorder.from_environment(
        "qwen", int(os.environ.get("LEAD_TIMING_RANK", "0")), args.device,
    )
    print("[vlm_service] model loaded", flush=True)

    fast_preprocess = os.environ.get("LEAD_QWEN_FAST_PREPROCESS", "1") == "1"
    prepared_prompt = prepare_p6_batch_prompt(processor) if fast_preprocess else None
    image_preprocess_device = (
        args.device
        if fast_preprocess and os.environ.get("LEAD_QWEN_IMAGE_PREPROCESS_DEVICE", "cpu") == "cuda"
        else None
    )
    parity_checked = False

    if os.path.exists(args.socket):
        os.remove(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    srv.listen(1)
    print(f"[vlm_service] listening on {args.socket} (prompt_mode={args.prompt_mode})", flush=True)

    while True:
        conn, _ = srv.accept()
        print("[vlm_service] client connected", flush=True)
        request_index = 0
        try:
            while True:
                header = _recv_exact(conn, 8)
                h, w = struct.unpack("<II", header)
                if h == 0:
                    count = w
                    timing = recorder.start(
                        "qwen_batch", index=request_index,
                        request_index=request_index, batch_size=count,
                    )
                    request_index += 1
                    with timing.phase("receive_payload"):
                        h, w = struct.unpack("<II", _recv_exact(conn, 8))
                        if not (1 <= count <= 256 and h == 384 and w == 1152):
                            raise ValueError(f"invalid batch request: {count}x{h}x{w}")
                        raw = _recv_exact(conn, count * h * w * 3)
                    with timing.phase("image_decode"):
                        images = np.frombuffer(raw, dtype=np.uint8).reshape(count, h, w, 3)
                        pil_images = [Image.fromarray(image, "RGB") for image in images]
                    if fast_preprocess and not parity_checked:
                        try:
                            reference = prepare_p6_batch_inputs(pil_images, processor)
                            candidate = prepare_p6_batch_inputs(
                                pil_images, processor, prompt=prepared_prompt,
                                image_device=image_preprocess_device,
                            )
                            check_p6_batch_input_parity(reference, candidate)
                            print("[vlm_service] fast P6 preprocessing parity passed", flush=True)
                        except Exception as exc:
                            if image_preprocess_device is not None:
                                print(
                                    f"[vlm_service] CUDA image preprocessing failed parity: "
                                    f"{exc}; retrying cached prompt on CPU", flush=True,
                                )
                                image_preprocess_device = None
                                try:
                                    candidate = prepare_p6_batch_inputs(
                                        pil_images, processor, prompt=prepared_prompt,
                                    )
                                    check_p6_batch_input_parity(reference, candidate)
                                except Exception as cpu_exc:
                                    exc = cpu_exc
                                    fast_preprocess = False
                            else:
                                fast_preprocess = False
                            if not fast_preprocess:
                                prepared_prompt = None
                                print(
                                    f"[vlm_service] fast P6 preprocessing disabled; "
                                    f"reference path retained: {exc}", flush=True,
                                )
                        parity_checked = True
                    feat = extract_vlm_hidden_batch(
                        pil_images,
                        model, processor, image_token_id, merge,
                        prompt_mode=args.prompt_mode, device=args.device, timing=timing,
                        prepared_prompt=prepared_prompt,
                        image_preprocess_device=image_preprocess_device,
                    )
                    _, fh, fw, fd = feat.shape
                    with timing.phase("send_reply"):
                        conn.sendall(struct.pack("<IIII", count, fh, fw, fd) + feat.tobytes())
                    timing.finish()
                    continue
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
