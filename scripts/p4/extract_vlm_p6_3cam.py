"""P6 verification: extract VLM features from the FULL 3-camera strip (not just the
front 1/3), to test whether Qwen + a light intent head can learn a BEV intent that lights
up SIDE arms (left/right turns) invisible to the front-only P5a.

Same as extract_vlm_features.py --prompt-mode drivable, but:
  - uses the whole 1152x384 strip (front-left -54.5, front 0, front-right +54.5)
    instead of cropping the middle 384 front camera;
  - a 3-camera drivable prompt.
Cache goes to a separate dir so it doesn't clobber the front-only P5 cache.

Run (qwenvl env), small verification subset:
    CUDA_VISIBLE_DEVICES=0 python scripts/p4/extract_vlm_p6_3cam.py \
        --manifest data/p4/manifest.jsonl --out data/p6/vlm_cache_3cam \
        --model /mmu_mllm_hdd_3/liuzihan08/vla/models/Qwen3-VL-4B-Instruct \
        --limit 4000
"""

from __future__ import annotations

import argparse
import json
import os


PROMPT_3CAM_DRIVABLE = (
    "You are the motion planner of a car. The image is a panorama stitched from three "
    "front-facing cameras: front-left, front, and front-right. Look across the whole "
    "panorama and identify every direction the car could drive from here -- all lanes and "
    "turns that are drivable ahead, including to the sides, without committing to one."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/p4/manifest.jsonl")
    ap.add_argument("--out", default="data/p6/vlm_cache_3cam")
    ap.add_argument("--model", default="/mmu_mllm_hdd_3/liuzihan08/vla/models/Qwen3-VL-4B-Instruct")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0, help="only first N assigned frames")
    args = ap.parse_args()

    import numpy as np
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

    done = skipped = bad = 0
    for e in entries:
        out_npy = os.path.join(args.out, e["scenario"], e["route"], e["frame"] + ".npy")
        if os.path.exists(out_npy):
            skipped += 1
            continue
        try:
            img = Image.open(e["src"]).convert("RGB")  # FULL strip, no crop
        except Exception as exc:  # noqa: BLE001
            bad += 1
            if bad <= 5:
                print(f"  [bad] {e['src']} ({exc})")
            continue

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": PROMPT_3CAM_DRIVABLE},
                {"type": "image", "image": img},
            ]}
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to("cuda:0")

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[args.layer][0]
        mask = inputs["input_ids"][0] == image_token_id
        img_hs = hs[mask]

        grid = inputs["image_grid_thw"][0].tolist()  # [t, h, w]
        h2, w2 = grid[1] // merge, grid[2] // merge
        if h2 * w2 != img_hs.shape[0]:
            bad += 1
            print(f"  [bad] token/grid mismatch {img_hs.shape[0]} vs {h2}x{w2} for {e['key']}")
            continue
        feat = img_hs.reshape(h2, w2, -1).to(torch.float16).cpu().numpy()

        os.makedirs(os.path.dirname(out_npy), exist_ok=True)
        np.save(out_npy, feat)
        done += 1
        if done <= 3 or done % 500 == 0:
            print(f"[shard {args.shard}] {done} done  {e['key']}  feat={feat.shape} "
                  f"({feat.nbytes/1024:.0f}KB, grid={grid})")

    print(f"[shard {args.shard}] finished: done={done} skipped={skipped} bad={bad}")


if __name__ == "__main__":
    main()
