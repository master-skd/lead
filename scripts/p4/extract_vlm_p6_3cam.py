"""Extract VLM features from the FULL 3-camera strip (not just the front 1/3).

Same as extract_vlm_features.py --prompt-mode drivable, but:
  - uses the whole 1152x384 strip (front-left -54.5, front 0, front-right +54.5)
    instead of cropping the middle 384 front camera;
  - a 3-camera drivable prompt.
Cache goes to a separate dir so it doesn't clobber the front-only P5 cache.
Existing files are validated before skipping; new files are published atomically.

Run (qwenvl env), full-frame manifest (parallelize with --num-shards/--shard):
    CUDA_VISIBLE_DEVICES=0 python scripts/p4/extract_vlm_p6_3cam.py \
        --manifest outputs/local_training/p5_stepB3a_v2_dense_data/manifest_stride1.jsonl \
        --out data/p6/vlm_cache_3cam \
        --model /mmu_mllm_hdd_3/liuzihan08/vla/models/Qwen3-VL-4B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


EXPECTED_CACHE_SHAPE = (12, 36, 2560)


def valid_cache_file(path: str, expected_shape: tuple[int, ...] = EXPECTED_CACHE_SHAPE) -> bool:
    """Check the header and payload length without reading a 2.1 MB feature into RAM."""
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return (
            isinstance(array, np.memmap)
            and array.shape == expected_shape
            and array.dtype == np.dtype("float16")
            and os.path.getsize(path) == array.offset + array.nbytes
        )
    except (OSError, ValueError, EOFError):
        return False


def atomic_save_cache(path: str, feature: np.ndarray) -> None:
    """Publish a complete cache file only after its temporary copy passes validation."""
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".vlm_", suffix=".npy.tmp", dir=parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, feature, allow_pickle=False)
        if not valid_cache_file(temporary):
            raise ValueError(f"invalid VLM feature written to {temporary}: {feature.shape} {feature.dtype}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def split_image_features(
    hidden: np.ndarray, input_ids: np.ndarray, image_grids: np.ndarray,
    image_token_id: int, merge: int,
) -> list[tuple[np.ndarray, list[int]]]:
    """Recover one image-token feature grid per padded conversation in a batch."""
    if hidden.shape[:2] != input_ids.shape or image_grids.shape != (hidden.shape[0], 3):
        raise ValueError("batched Qwen hidden states, tokens and image grids do not align")
    features = []
    for index, grid_array in enumerate(image_grids):
        grid = grid_array.tolist()
        h2, w2 = grid[1] // merge, grid[2] // merge
        image_hidden = hidden[index][input_ids[index] == image_token_id]
        if h2 * w2 != image_hidden.shape[0]:
            raise ValueError(
                f"image token/grid mismatch in batch slot {index}: "
                f"{image_hidden.shape[0]} vs {h2}x{w2}"
            )
        features.append((image_hidden.reshape(h2, w2, -1), grid))
    return features


PROMPT_3CAM_DRIVABLE = (
    "You are the motion planner of a car. The image is a panorama stitched from three "
    "front-facing cameras: front-left, front, and front-right. Look across the whole "
    "panorama and identify every direction the car could drive from here -- all lanes and "
    "turns that are drivable ahead, including to the sides, without committing to one."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default="outputs/local_training/p5_stepB3a_v2_dense_data/manifest_stride1.jsonl",
    )
    ap.add_argument("--out", default="data/p6/vlm_cache_3cam")
    ap.add_argument("--model", default="/mmu_mllm_hdd_3/liuzihan08/vla/models/Qwen3-VL-4B-Instruct")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0, help="only first N assigned frames")
    ap.add_argument("--batch-size", type=int, default=1, help="images per Qwen forward")
    ap.add_argument("--num-workers", type=int, default=4, help="parallel JPEG readers")
    ap.add_argument("--prefetch-batches", type=int, default=4, help="JPEG batches queued ahead")
    args = ap.parse_args()
    if min(args.batch_size, args.num_workers, args.prefetch_batches) < 1:
        ap.error("batch-size, num-workers and prefetch-batches must be positive")
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        ap.error("shard must be in [0, num-shards)")

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    os.makedirs(args.out, exist_ok=True)
    entries = [json.loads(l) for l in open(args.manifest)]
    entries = [e for i, e in enumerate(entries) if i % args.num_shards == args.shard]
    if args.limit > 0:
        entries = entries[: args.limit]
    print(f"[shard {args.shard}/{args.num_shards}] frames: {len(entries)}  (FULL 3-cam strip)")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()
    image_token_id = model.config.image_token_id
    merge = getattr(model.config.vision_config, "spatial_merge_size", 2)

    def load_rgb(entry):
        try:
            with Image.open(entry["src"]) as source:
                return source.convert("RGB"), None  # FULL strip, no crop
        except Exception as exc:  # noqa: BLE001
            return None, exc

    done = skipped = repaired = bad = 0
    started = time.monotonic()
    window_size = args.batch_size * args.prefetch_batches
    with ThreadPoolExecutor(max_workers=args.num_workers) as readers:
        for start in range(0, len(entries), window_size):
            pending = []
            for entry in entries[start : start + window_size]:
                out_npy = os.path.join(
                    args.out, entry["scenario"], entry["route"], entry["frame"] + ".npy"
                )
                if os.path.exists(out_npy):
                    if valid_cache_file(out_npy):
                        skipped += 1
                        continue
                    repaired += 1
                    print(f"[shard {args.shard}] invalid existing cache, re-extracting: {out_npy}")
                pending.append((entry, out_npy, readers.submit(load_rgb, entry)))

            for offset in range(0, len(pending), args.batch_size):
                batch = []
                for entry, out_npy, future in pending[offset : offset + args.batch_size]:
                    image, error = future.result()
                    if error is not None:
                        bad += 1
                        if bad <= 5:
                            print(f"  [bad] {entry['src']} ({error})")
                        continue
                    batch.append((entry, out_npy, image))
                if not batch:
                    continue

                conversations = [
                    [{"role": "user", "content": [
                        {"type": "text", "text": PROMPT_3CAM_DRIVABLE},
                        {"type": "image", "image": image},
                    ]}]
                    for _, _, image in batch
                ]
                inputs = processor.apply_chat_template(
                    conversations[0] if len(batch) == 1 else conversations,
                    tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                    processor_kwargs={"padding": len(batch) > 1},
                ).to("cuda:0")
                if inputs["image_grid_thw"].shape[0] != len(batch):
                    raise ValueError("Qwen processor did not return one image grid per frame")

                with torch.inference_mode():
                    out = model(**inputs, output_hidden_states=True, use_cache=False)
                features = split_image_features(
                    out.hidden_states[args.layer].to(torch.float16).cpu().numpy(),
                    inputs["input_ids"].cpu().numpy(),
                    inputs["image_grid_thw"].cpu().numpy(),
                    image_token_id, merge,
                )
                for (entry, out_npy, _), (feature, grid) in zip(batch, features):
                    if feature.shape != EXPECTED_CACHE_SHAPE or not np.isfinite(feature).all():
                        raise ValueError(
                            f"unexpected VLM feature for {entry['key']}: {feature.shape} {feature.dtype}"
                        )
                    atomic_save_cache(out_npy, feature)
                    done += 1
                    if done <= 3 or done % 500 == 0:
                        rate = done / max(time.monotonic() - started, 1e-6)
                        print(
                            f"[shard {args.shard}] {done} done, {skipped} skipped "
                            f"({rate:.2f} new frame/s)  {entry['key']}  "
                            f"feat={feature.shape} ({feature.nbytes/1024:.0f}KB, grid={grid})"
                        )
                del out, inputs, features

    elapsed = time.monotonic() - started
    print(
        f"[shard {args.shard}] finished: done={done} skipped={skipped} "
        f"repaired={repaired} bad={bad} elapsed={elapsed:.1f}s "
        f"new_rate={done / max(elapsed, 1e-6):.2f} frame/s"
    )


if __name__ == "__main__":
    main()
