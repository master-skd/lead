"""Compare a deployable CenterNet constant-velocity gate with GT actor oracle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _load_predicted_boxes(path: Path) -> tuple[dict[str, np.ndarray], str]:
    boxes_by_key = {}
    source = None
    paths = sorted(path.glob("predicted_boxes_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no predicted actor cache in {path}")
    for shard_path in paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            checkpoint = str(shard["source_checkpoint"])
            if source is None:
                source = checkpoint
            elif source != checkpoint:
                raise ValueError("predicted actor shards use different checkpoints")
            for key, boxes in zip(shard["keys"], shard["boxes"], strict=True):
                if str(key) in boxes_by_key:
                    raise ValueError(f"duplicate predicted actor key: {key}")
                boxes_by_key[str(key)] = boxes
    return boxes_by_key, source


def _load_sample_features(
    path: Path, boxes_by_key: dict[str, np.ndarray]
) -> list[dict]:
    frames = []
    matched = set()
    for shard_path in sorted(path.glob("velocity_features_*.npz")):
        with np.load(shard_path, allow_pickle=False) as shard:
            keys = shard["keys"]
            indices = [i for i, key in enumerate(keys) if str(key) in boxes_by_key]
            if not indices:
                continue
            fields = (
                "candidate_states",
                "candidate_valid",
                "candidate_velocity",
                "collision",
                "imitation_error",
            )
            selected = {name: shard[name][indices] for name in fields}
            for row, index in enumerate(indices):
                key = str(keys[index])
                matched.add(key)
                frames.append(
                    {"key": key, "boxes": boxes_by_key[key]}
                    | {name: value[row] for name, value in selected.items()}
                )
    if matched != set(boxes_by_key):
        raise ValueError(
            f"{len(set(boxes_by_key) - matched)} predicted frames lack labels"
        )
    return frames


def _summarize(records: list[dict], threshold: float) -> dict:
    n = len(records)
    raw = np.asarray([r["raw_collision"] for r in records], dtype=bool)
    predicted_raw = np.asarray(
        [r["predicted_raw_collision"] for r in records], dtype=bool
    )
    selected = np.asarray([r["selected_collision"] for r in records], dtype=bool)
    oracle = np.asarray([r["oracle_collision"] for r in records], dtype=bool)
    switched = np.asarray([r["selected_index"] != 0 for r in records], dtype=bool)
    oracle_switched = np.asarray([r["oracle_index"] != 0 for r in records], dtype=bool)
    raw_mae = np.asarray([r["raw_mae"] for r in records], dtype=np.float32)
    selected_mae = np.asarray([r["selected_mae"] for r in records], dtype=np.float32)
    raw_progress = np.asarray([r["raw_progress"] for r in records], dtype=np.float32)
    selected_progress = np.asarray(
        [r["selected_progress"] for r in records], dtype=np.float32
    )
    candidate_tp = sum(r["candidate_tp"] for r in records)
    candidate_fp = sum(r["candidate_fp"] for r in records)
    candidate_fn = sum(r["candidate_fn"] for r in records)
    return {
        "score_threshold": threshold,
        "n_frames": n,
        "predicted_actor_count_mean": float(
            np.mean([r["predicted_actor_count"] for r in records])
        ),
        "no_predicted_actor_rate": float(
            np.mean([r["predicted_actor_count"] == 0 for r in records])
        ),
        "raw_gt_collision_rate": float(raw.mean()),
        "oracle_gt_collision_rate": float(oracle.mean()),
        "oracle_switch_rate": float(oracle_switched.mean()),
        "predicted_raw_collision_rate": float(predicted_raw.mean()),
        "raw_collision_recall": float((raw & predicted_raw).sum() / max(raw.sum(), 1)),
        "raw_collision_precision": float(
            (raw & predicted_raw).sum() / max(predicted_raw.sum(), 1)
        ),
        "raw_false_alarm_rate": float((~raw & predicted_raw).mean()),
        "candidate_collision_recall": candidate_tp
        / max(candidate_tp + candidate_fn, 1),
        "candidate_collision_precision": candidate_tp
        / max(candidate_tp + candidate_fp, 1),
        "selected_gt_collision_rate": float(selected.mean()),
        "rescue_rate": float((raw & ~selected).mean()),
        "introduced_collision_rate": float((~raw & selected).mean()),
        "switch_rate": float(switched.mean()),
        "false_switch_rate": float((~raw & switched).mean()),
        "raw_mae_mps": float(raw_mae.mean()),
        "selected_mae_mps": float(selected_mae.mean()),
        "raw_progress_m": float(raw_progress.mean()),
        "selected_progress_m": float(selected_progress.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--predicted-box-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--score-thresholds", default="0.3,0.5,0.7")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--safety-margin", type=float, default=0.2)
    args = parser.parse_args()
    thresholds = [float(item) for item in args.score_thresholds.split(",")]
    if not thresholds or any(not 0 < value < 1 for value in thresholds):
        parser.error("score thresholds must lie in (0, 1)")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from lead.tfv6.predicted_actor_velocity_gate import (
        extrapolate_detected_actors,
        predicted_candidate_collisions,
        select_safe_slowdown,
    )

    predicted_box_dir = Path(args.predicted_box_dir)
    boxes_by_key, checkpoint = _load_predicted_boxes(predicted_box_dir)
    frames = _load_sample_features(Path(args.feature_cache_dir), boxes_by_key)
    output = {
        "source_checkpoint": checkpoint,
        "feature_cache": str(Path(args.feature_cache_dir).resolve()),
        "predicted_box_cache": str(predicted_box_dir.resolve()),
        "actor_motion_model": "constant speed along detected yaw over eight 0.25 s steps",
        "selection": "keep raw unless predicted collision; fastest predicted-safe slowdown",
        "safety_margin_m": args.safety_margin,
        "nms_iou_threshold": args.nms_iou_threshold,
        "threshold_sweep": [],
    }
    frame_details = {"keys": np.asarray([frame["key"] for frame in frames])}
    for threshold in thresholds:
        records = []
        for frame in tqdm(frames, desc=f"predicted actors >= {threshold:.2f}"):
            actors = extrapolate_detected_actors(
                frame["boxes"],
                score_threshold=threshold,
                nms_iou_threshold=args.nms_iou_threshold,
            )
            valid = frame["candidate_valid"].astype(bool)
            ground_truth = frame["collision"].astype(bool)
            predicted = predicted_candidate_collisions(
                frame["candidate_states"],
                valid,
                actors,
                safety_margin_m=args.safety_margin,
            )
            velocity = frame["candidate_velocity"]
            selected_index = select_safe_slowdown(valid, predicted, velocity)
            oracle_index = select_safe_slowdown(valid, ground_truth, velocity)
            progress = velocity.astype(np.float32).clip(min=0).sum(1) * 0.25
            records.append(
                {
                    "raw_collision": bool(ground_truth[0]),
                    "predicted_raw_collision": bool(predicted[0]),
                    "selected_collision": bool(ground_truth[selected_index]),
                    "oracle_collision": bool(ground_truth[oracle_index]),
                    "selected_index": selected_index,
                    "oracle_index": oracle_index,
                    "candidate_tp": int((valid & predicted & ground_truth).sum()),
                    "candidate_fp": int((valid & predicted & ~ground_truth).sum()),
                    "candidate_fn": int((valid & ~predicted & ground_truth).sum()),
                    "predicted_actor_count": actors.num_actors,
                    "raw_mae": float(frame["imitation_error"][0]),
                    "selected_mae": float(frame["imitation_error"][selected_index]),
                    "raw_progress": float(progress[0]),
                    "selected_progress": float(progress[selected_index]),
                }
            )
        summary = _summarize(records, threshold)
        output["threshold_sweep"].append(summary)
        tag = f"score_{threshold:.2f}".replace(".", "p")
        for field in (
            "predicted_raw_collision",
            "selected_collision",
            "selected_index",
            "oracle_index",
            "predicted_actor_count",
        ):
            frame_details[f"{tag}_{field}"] = np.asarray(
                [record[field] for record in records]
            )
        print(
            f"threshold={threshold:.2f} raw={summary['raw_gt_collision_rate']:.2%} "
            f"oracle={summary['oracle_gt_collision_rate']:.2%} "
            f"predicted={summary['selected_gt_collision_rate']:.2%} "
            f"recall={summary['raw_collision_recall']:.1%} "
            f"false-switch={summary['false_switch_rate']:.2%} "
            f"MAE={summary['raw_mae_mps']:.3f}"
            f"->{summary['selected_mae_mps']:.3f} m/s"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    np.savez_compressed(out.with_suffix(".frames.npz"), **frame_details)
    print(f"wrote {out} and {out.with_suffix('.frames.npz')}")


if __name__ == "__main__":
    main()
