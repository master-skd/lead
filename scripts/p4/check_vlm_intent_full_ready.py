"""Reject full-intent training until all exact-frame features and labels are complete."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def completed_frames(log_dir: Path, shards: int, kind: str) -> int:
    total = 0
    for shard in range(shards):
        path = log_dir / f"shard_{shard}.log"
        text = path.read_text()
        header = re.search(rf"\[shard {shard}/{shards}\] frames: (\d+)", text)
        summary = re.search(r"finished: done=(\d+) skipped=(\d+)(?: repaired=\d+)? bad=(\d+)", text)
        if header is None or summary is None:
            raise ValueError(f"{kind} shard {shard} has not finished: {path}")
        frames = int(header.group(1))
        done, skipped, bad = map(int, summary.groups())
        if done + skipped != frames or bad:
            raise ValueError(
                f"{kind} shard {shard} incomplete: expected={frames} "
                f"done={done} skipped={skipped} bad={bad}"
            )
        total += frames
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-logs", type=Path,
                        default=Path("outputs/local_training/vlm_cache_3cam_full_logs"))
    parser.add_argument("--lanegraph-logs", type=Path,
                        default=Path("outputs/local_training/lanegraph_intent_full_logs"))
    parser.add_argument("--lanegraph-shards", type=int, default=4)
    parser.add_argument("--split-metadata", type=Path,
                        default=Path("outputs/local_training/p5_stepB3a_v2_dense_data/route_split/metadata.json"))
    args = parser.parse_args()

    metadata = json.loads(args.split_metadata.read_text())
    if metadata["route_overlap"] or metadata["train_frames"] + metadata["heldout_frames"] != metadata["total_frames"]:
        raise ValueError("train/heldout route split is invalid")
    expected = metadata["total_frames"]
    vlm_frames = completed_frames(args.vlm_logs, 8, "VLM")
    lanegraph_frames = completed_frames(args.lanegraph_logs, args.lanegraph_shards, "lane-graph")
    if vlm_frames != expected or lanegraph_frames != expected:
        raise ValueError(
            f"manifest/asset mismatch: split={expected} VLM={vlm_frames} lane-graph={lanegraph_frames}"
        )
    print(
        f"full intent assets ready: total={expected}, "
        f"train={metadata['train_frames']}, heldout={metadata['heldout_frames']}, route_overlap=0"
    )


if __name__ == "__main__":
    main()
