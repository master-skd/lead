"""P5 B2a: precompute anchors for every frame (offline, since the P6 intent head is FROZEN
-> pred blob is deterministic -> anchors are deterministic). Training then reads tiny cached
anchor arrays instead of running skimage skeletonization in the loop (which is CPU-only,
per-frame, un-batchable, and would throttle throughput).

For each cached 3-cam vlm_hidden: run the frozen P6 VLMIntentDecoder -> sigmoid blob ->
extract_anchors_from_blob -> save (K_MAX, 5) float16 = [valid, angle, reach, tip_row,
tip_col] to <out>/<scenario>/<route>/<frame>.npy (a few hundred bytes/frame).

Shardable across GPUs. Run (lead env):
    for i in 0..7: CUDA_VISIBLE_DEVICES=$i python scripts/p4/precompute_anchors.py \
        --intent-ckpt outputs/local_training/vlm_intent_p6_3cam/model_0014.pth \
        --vlm-cache data/p6/vlm_cache_3cam --out data/p6/anchor_cache \
        --num-shards 8 --shard $i &
"""

from __future__ import annotations

import argparse
import glob
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intent-ckpt", default="outputs/local_training/vlm_intent_p6_3cam/model_0014.pth")
    ap.add_argument("--vlm-cache", default="data/p6/vlm_cache_3cam")
    ap.add_argument("--out", default="data/p6/anchor_cache")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import torch
    from lead.tfv6.vlm_intent_decoder import VLMIntentDecoder
    from lead.tfv6.anchor_extraction import extract_anchors_from_blob, K_MAX
    from lead.training.config_training import TrainingConfig

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = TrainingConfig({"use_multimodal_intent": True}, raise_error_on_missing_key=False)
    ppm = cfg.pixels_per_meter
    row_ego = int((0 - cfg.min_y_meter) * ppm)   # 160
    col_ego = int((0 - cfg.min_x_meter) * ppm)   # 128

    decoder = VLMIntentDecoder(cfg).to(device)
    decoder.load_state_dict(torch.load(args.intent_ckpt, map_location=device, weights_only=True)["model"])
    decoder.eval().requires_grad_(False)

    # enumerate all cached vlm_hidden npy, shard by index
    npys = sorted(glob.glob(os.path.join(args.vlm_cache, "*", "*", "*.npy")))
    npys = [p for i, p in enumerate(npys) if i % args.num_shards == args.shard]
    if args.limit > 0:
        npys = npys[: args.limit]
    print(f"[shard {args.shard}/{args.num_shards}] frames: {len(npys)}", flush=True)

    done = skipped = 0
    for p in npys:
        parts = p.split("/")
        scenario, route, frame = parts[-3], parts[-2], parts[-1][:-4]
        out_npy = os.path.join(args.out, scenario, route, frame + ".npy")
        if os.path.exists(out_npy):
            skipped += 1
            continue
        vh = torch.from_numpy(np.load(p)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            blob = torch.sigmoid(decoder(vh)).cpu().numpy()[0, 0]  # (H,W)
        anchors, endpoints = extract_anchors_from_blob(blob, ppm, row_ego, col_ego)
        # pack (K_MAX,5) = [valid, angle, reach, tip_row, tip_col]
        packed = np.concatenate([anchors, endpoints], axis=1).astype(np.float16)  # (K_MAX,5)
        os.makedirs(os.path.dirname(out_npy), exist_ok=True)
        np.save(out_npy, packed)
        done += 1
        if done <= 3 or done % 2000 == 0:
            k = int(anchors[:, 0].sum())
            print(f"[shard {args.shard}] {done} done  {scenario}/{frame}  K={k}", flush=True)

    print(f"[shard {args.shard}] finished: done={done} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
