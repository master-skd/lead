"""Speed-only vLLM adapter for the existing batched VLM Unix socket protocol.

The extracted features are deliberately *not* assumed equivalent to the HF
features. Do not use checkpoints trained through this adapter for evaluation.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import struct
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from lead.inference.vlm_feature_extractor import PROMPT_3CAM_DRIVABLE
from lead.inference.vlm_service import _recv_exact
from lead.common.stage_timing import StageTimingRecorder


def encode_image(image: np.ndarray) -> str:
    stream = BytesIO()
    Image.fromarray(image, "RGB").save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def extract_one(image: np.ndarray | Path, *, url: str, model: str, shared_dir: Path,
                image_token_id: int) -> np.ndarray:
    from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector as connector

    image_url = image.as_uri() if isinstance(image, Path) else encode_image(image)
    body = {
        "model": model, "max_tokens": 1, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT_3CAM_DRIVABLE},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}],
        "kv_transfer_params": {"include_output_tokens": False},
    }
    request = Request(url.rstrip("/") + "/v1/chat/completions",
                      data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=300) as response:
        reply = json.load(response)
    relative = (reply.get("kv_transfer_params") or {}).get("hidden_states_path")
    if not relative:
        raise RuntimeError("vLLM did not return hidden_states_path")
    path = Path(relative).resolve()
    if not path.is_relative_to(shared_dir) or path.suffix != ".safetensors":
        raise ValueError(f"unsafe vLLM hidden-state path: {path}")
    saved = connector.load_hidden_states(str(path))
    try:
        ids = saved["token_ids"].cpu().numpy().reshape(-1)
        hidden = saved["hidden_states"].float().cpu().numpy()
    finally:
        connector.cleanup_hidden_states(str(path))
    selected = hidden[ids == image_token_id, 0]
    if selected.shape != (432, 2560):
        raise ValueError(f"unexpected vLLM image feature shape: {selected.shape}")
    return selected.reshape(12, 36, 2560).astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--served-model", default="p6-vllm-pilot")
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    args.shared_dir = args.shared_dir.resolve()
    args.media_root = args.media_root.resolve()
    config = json.loads((args.model_dir / "config.json").read_text())
    image_token_id = int(config["image_token_id"])
    recorder = StageTimingRecorder.from_environment(
        "vllm_bridge", int(os.environ.get("LEAD_TIMING_RANK", "0")),
    )

    if args.socket.exists():
        raise FileExistsError(args.socket)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(args.socket))
        server.listen(1)
        print(f"[vllm_bridge] listening on {args.socket}", flush=True)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            request_index = 0
            while True:
                conn, _ = server.accept()
                with conn:
                    print("[vllm_bridge] client connected", flush=True)
                    try:
                        while True:
                            h, w = struct.unpack("<II", _recv_exact(conn, 8))
                            if h != 0:
                                raise ValueError("speed-only bridge requires batched requests")
                            count = w
                            timing = recorder.start("vllm_batch", index=request_index,
                                                    request_index=request_index, batch_size=count)
                            request_index += 1
                            with timing.phase("receive_payload"):
                                h, w = struct.unpack("<II", _recv_exact(conn, 8))
                                if not (1 <= count <= 256):
                                    raise ValueError(f"invalid batch size: {count}")
                                if (h, w) == (0, 1):
                                    images = []
                                    for _ in range(count):
                                        length, = struct.unpack("<I", _recv_exact(conn, 4))
                                        if not (1 <= length <= 65535):
                                            raise ValueError(f"invalid path length: {length}")
                                        path = Path(_recv_exact(conn, length).decode("utf-8")).resolve()
                                        if not path.is_relative_to(args.media_root) or not path.is_file():
                                            raise ValueError(f"image is outside CARLA root or missing: {path}")
                                        images.append(path)
                                elif (h, w) == (384, 1152):
                                    images = np.frombuffer(_recv_exact(conn, count * h * w * 3),
                                                           dtype=np.uint8).reshape(count, h, w, 3)
                                else:
                                    raise ValueError(f"invalid batch encoding: {count}x{h}x{w}")
                            with timing.phase("vllm_requests"):
                                features = np.stack(list(pool.map(
                                    lambda image: extract_one(
                                        image, url=args.base_url, model=args.served_model,
                                        shared_dir=args.shared_dir, image_token_id=image_token_id,
                                    ), images,
                                )))
                            with timing.phase("send_reply"):
                                conn.sendall(struct.pack("<IIII", count, 12, 36, 2560)
                                             + features.tobytes())
                            timing.finish()
                    except (ConnectionError, BrokenPipeError):
                        print("[vllm_bridge] client disconnected", flush=True)
    # The benchmark launcher owns this socket path and removes it on exit.


if __name__ == "__main__":
    main()
