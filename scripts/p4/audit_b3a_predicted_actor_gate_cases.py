"""Explain predicted-actor gate misses using held-out GT labels (audit only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.p4.eval_b3a_predicted_actor_gate import (
    _load_predicted_boxes,
    _load_sample_features,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--predicted-box-dir", required=True)
    parser.add_argument("--frame-details", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--examples-per-group", type=int, default=20)
    args = parser.parse_args()

    boxes, _ = _load_predicted_boxes(Path(args.predicted_box_dir))
    features = _load_sample_features(Path(args.feature_cache_dir), boxes)
    collision_by_key = {
        frame["key"]: bool(frame["collision"][0]) for frame in features
    }
    tag = f"score_{args.threshold:.2f}".replace(".", "p")
    with np.load(args.frame_details, allow_pickle=False) as details:
        keys = details["keys"].tolist()
        predicted_raw = details[f"{tag}_predicted_raw_collision"].astype(bool)
        selected_collision = details[f"{tag}_selected_collision"].astype(bool)
        selected_index = details[f"{tag}_selected_index"]
        oracle_index = details[f"{tag}_oracle_index"]
        actor_count = details[f"{tag}_predicted_actor_count"]

    if len(set(keys)) != len(keys) or set(keys) != set(collision_by_key):
        raise ValueError("frame-detail keys and GT feature keys do not match")
    raw_collision = np.asarray([collision_by_key[key] for key in keys])
    possible = raw_collision & (oracle_index != 0)
    switched = selected_index != 0
    groups = {
        "oracle_rescued": possible & ~selected_collision,
        "oracle_missed_raw_not_flagged": possible & ~predicted_raw,
        "oracle_flagged_no_switch": possible & predicted_raw & ~switched,
        "oracle_switched_but_still_collision": possible & switched & selected_collision,
        "not_oracle_rescuable": raw_collision & ~possible,
        "raw_safe_false_switch": ~raw_collision & switched,
        "introduced_collision": ~raw_collision & selected_collision,
    }
    report = {
        "threshold": args.threshold,
        "n_frames": len(keys),
        "raw_gt_collision_count": int(raw_collision.sum()),
        "oracle_rescuable_count": int(possible.sum()),
        "groups": {},
    }
    for name, mask in groups.items():
        indices = np.flatnonzero(mask)
        report["groups"][name] = {
            "count": len(indices),
            "examples": [
                {
                    "key": keys[i],
                    "raw_predicted_collision": bool(predicted_raw[i]),
                    "selected_index": int(selected_index[i]),
                    "oracle_index": int(oracle_index[i]),
                    "predicted_actor_count": int(actor_count[i]),
                }
                for i in indices[: args.examples_per_group]
            ],
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    for name, group in report["groups"].items():
        print(f"  {name}: {group['count']}")


if __name__ == "__main__":
    main()
