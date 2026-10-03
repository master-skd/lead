"""Compare a batched extraction pilot against existing single-frame Qwen features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/local_training/p5_stepB3a_v2_dense_data/manifest_stride1.jsonl"),
    )
    parser.add_argument("--reference", type=Path, default=Path("data/p6/vlm_cache_3cam"))
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()

    maes = []
    cosines = []
    with args.manifest.open() as stream:
        for index, line in enumerate(stream):
            if index % args.num_shards != args.shard:
                continue
            entry = json.loads(line)
            relative = Path(entry["scenario"]) / entry["route"] / f"{entry['frame']}.npy"
            reference = np.load(args.reference / relative, allow_pickle=False).astype(np.float32)
            pilot = np.load(args.pilot / relative, allow_pickle=False).astype(np.float32)
            if reference.shape != pilot.shape:
                raise ValueError(f"shape mismatch for {relative}: {reference.shape} vs {pilot.shape}")
            maes.append(float(np.abs(reference - pilot).mean()))
            ref_flat, pilot_flat = reference.reshape(-1), pilot.reshape(-1)
            cosine = float(np.dot(ref_flat, pilot_flat) / (
                np.linalg.norm(ref_flat) * np.linalg.norm(pilot_flat) + 1e-12
            ))
            cosines.append(cosine)
            if len(maes) >= args.limit:
                break

    if not maes:
        raise ValueError("no frames compared")
    print(
        f"frames={len(maes)} feature MAE mean/max={np.mean(maes):.6f}/{np.max(maes):.6f} "
        f"cosine mean/min={np.mean(cosines):.6f}/{np.min(cosines):.6f}"
    )


if __name__ == "__main__":
    main()
