"""Speed-only Qwen3-VL token-pooling service for P6 joint training.

One offline vLLM instance serves one training rank through the existing Unix
socket protocol. The frozen HF intent head is not feature-parity validated for
these outputs; checkpoints produced with this service are benchmark-only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import struct

import numpy as np
from PIL import Image
import torch

from lead.common.stage_timing import StageTimingRecorder
from lead.inference.vlm_feature_extractor import PROMPT_3CAM_DRIVABLE
from lead.inference.vlm_service import _recv_exact


def receive_paths(conn: socket.socket, count: int, media_root: Path) -> list[Path]:
    h, w = struct.unpack("<II", _recv_exact(conn, 8))
    if (h, w) != (0, 1) or not (1 <= count <= 256):
        raise ValueError(f"pooling service requires 1..256 image paths: {count}, {h}, {w}")
    paths = []
    for _ in range(count):
        length, = struct.unpack("<I", _recv_exact(conn, 4))
        if not (1 <= length <= 65535):
            raise ValueError(f"invalid path length: {length}")
        path = Path(_recv_exact(conn, length).decode("utf-8")).resolve()
        if not path.is_relative_to(media_root) or not path.is_file():
            raise ValueError(f"image outside CARLA root or missing: {path}")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=524288)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = parser.parse_args()
    args.media_root = args.media_root.resolve()
    if args.max_num_seqs < 1:
        parser.error("max-num-seqs must be positive")

    from transformers import AutoProcessor
    from vllm import LLM
    from vllm.config import PoolerConfig

    config = json.loads((args.model_dir / "config.json").read_text())
    image_token_id = int(config["image_token_id"])
    processor = AutoProcessor.from_pretrained(str(args.model_dir), trust_remote_code=True)
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "text", "text": PROMPT_3CAM_DRIVABLE},
            {"type": "image"},
        ]}], tokenize=False, add_generation_prompt=True,
    )
    llm = LLM(
        model=str(args.model_dir), runner="pooling", convert="embed",
        dtype="bfloat16", max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        mm_processor_kwargs={"max_pixels": args.max_pixels},
        limit_mm_per_prompt={"image": 1},
        enable_chunked_prefill=False, enable_prefix_caching=False,
        pooler_config=PoolerConfig(
            task="token_embed", pooling_type="ALL", use_activation=False,
        ),
    )
    recorder = StageTimingRecorder.from_environment(
        "qwen_pooling", int(os.environ.get("LEAD_TIMING_RANK", "0")),
    )
    if args.socket.exists():
        raise FileExistsError(args.socket)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(args.socket))
        server.listen(1)
        print(f"[vllm_pooling] listening on {args.socket}", flush=True)
        request_index = 0
        while True:
            conn, _ = server.accept()
            with conn:
                print("[vllm_pooling] client connected", flush=True)
                try:
                    while True:
                        h, count = struct.unpack("<II", _recv_exact(conn, 8))
                        if h != 0:
                            raise ValueError("pooling service requires batched path requests")
                        timing = recorder.start(
                            "pooling_batch", index=request_index,
                            request_index=request_index, batch_size=count,
                        )
                        request_index += 1
                        with timing.phase("receive_paths"):
                            paths = receive_paths(conn, count, args.media_root)
                        if count > args.max_num_seqs:
                            raise ValueError(f"batch {count} exceeds max-num-seqs {args.max_num_seqs}")
                        with timing.phase("image_decode"):
                            images = []
                            for path in paths:
                                with Image.open(path) as image:
                                    images.append(image.convert("RGB"))
                        with timing.phase("pooling_encode"):
                            outputs = llm.encode(
                                [{"prompt": prompt, "multi_modal_data": {"image": image}}
                                 for image in images],
                                pooling_task="token_embed", use_tqdm=False,
                            )
                        with timing.phase("extract_and_to_cpu"):
                            if len(outputs) != count:
                                raise ValueError(f"vLLM returned {len(outputs)} of {count} outputs")
                            features = []
                            for output in outputs:
                                ids = np.asarray(output.prompt_token_ids)
                                hidden = output.outputs.data
                                if hidden.ndim != 2 or hidden.shape[0] != len(ids):
                                    raise ValueError("pooling hidden/token length mismatch")
                                mask = torch.as_tensor(ids == image_token_id, device=hidden.device)
                                selected = hidden[mask]
                                if selected.shape != (432, 2560):
                                    raise ValueError(f"unexpected image feature shape: {tuple(selected.shape)}")
                                features.append(selected.reshape(12, 36, 2560).to(
                                    dtype=torch.float16,
                                ).cpu().numpy())
                            result = np.stack(features)
                        with timing.phase("send_reply"):
                            conn.sendall(struct.pack("<IIII", count, 12, 36, 2560)
                                         + result.tobytes())
                        timing.finish()
                except (ConnectionError, BrokenPipeError):
                    print("[vllm_pooling] client disconnected", flush=True)


if __name__ == "__main__":
    main()
