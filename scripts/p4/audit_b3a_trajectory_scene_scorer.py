"""Audit B3a-v2 score distributions and counterfactual selections on held-out data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _sample_features(cache_dir: Path, samples_per_shard: int) -> dict[str, np.ndarray]:
    fields = (
        "scene_tokens",
        "candidate_states",
        "candidate_valid",
        "collision",
        "imitation_error",
        "label_collision_free",
        "label_ttc",
        "label_progress",
        "label_comfort",
        "label_imitation",
    )
    parts = {name: [] for name in fields}
    paths = sorted(cache_dir.glob("velocity_features_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no feature shards in {cache_dir}")
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            length = len(shard["candidate_valid"])
            count = min(length, samples_per_shard) if samples_per_shard else length
            indices = np.linspace(0, length - 1, count, dtype=np.int64)
            for name in fields:
                parts[name].append(shard[name][indices])
    return {name: np.concatenate(values) for name, values in parts.items()}


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float32)
    if not len(values):
        return {"count": 0}
    quantiles = np.quantile(values, (0, 0.1, 0.5, 0.9, 1))
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "min": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "max": float(quantiles[4]),
    }


def _within_frame_safety_ranking(
    score: np.ndarray, valid: np.ndarray, collision: np.ndarray
) -> dict[str, float | int]:
    correct = tied = pairs = mixed_frames = 0
    for row in range(len(valid)):
        safe_scores = score[row, valid[row] & ~collision[row]]
        unsafe_scores = score[row, valid[row] & collision[row]]
        if not len(safe_scores) or not len(unsafe_scores):
            continue
        mixed_frames += 1
        difference = safe_scores[:, None] - unsafe_scores[None, :]
        correct += int((difference > 0).sum())
        tied += int((difference == 0).sum())
        pairs += difference.size
    return {
        "mixed_frames": mixed_frames,
        "pairs": pairs,
        "safe_above_unsafe": (correct + 0.5 * tied) / max(pairs, 1),
    }


def _audit_scores(
    arrays: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    collision_free_threshold: float,
    ttc_threshold: float,
) -> dict:
    valid = arrays["candidate_valid"].astype(bool)
    collision = arrays["collision"].astype(bool)
    imitation_error = arrays["imitation_error"].astype(np.float32)
    n = len(valid)
    rows = np.arange(n)
    feasible = (
        valid
        & (scores["collision_free"] >= collision_free_threshold)
        & (scores["ttc"] >= ttc_threshold)
    )
    utility = (
        0.65 * scores["progress"]
        + 0.25 * scores["comfort"]
        + 0.10 * scores["imitation"]
    )
    preferred = np.where(feasible, utility, -np.inf).argmax(1)
    safety = 0.7 * scores["collision_free"] + 0.3 * scores["ttc"]
    fallback_index = np.where(valid, safety, -np.inf).argmax(1)
    has_feasible = feasible.any(1)
    selected = np.where(has_feasible, preferred, fallback_index)
    raw_collision = collision[:, 0]
    selected_collision = collision[rows, selected]
    safe_candidate = (valid & ~collision).any(1)
    safe_alternative = (valid[:, 1:] & ~collision[:, 1:]).any(1)
    head_distributions = {}
    for head, score in scores.items():
        head_distributions[head] = {
            "all_valid": _distribution(score[valid]),
            "raw": _distribution(score[:, 0]),
            "selected": _distribution(score[rows, selected]),
            "collision": _distribution(score[valid & collision]),
            "collision_free": _distribution(score[valid & ~collision]),
            "target": _distribution(arrays[f"label_{head}"][valid]),
            "within_frame_span": _distribution(
                np.where(valid, score, -np.inf).max(1)
                - np.where(valid, score, np.inf).min(1)
            ),
        }
    head_distributions["collision_free"]["within_frame_safety_ranking"] = (
        _within_frame_safety_ranking(scores["collision_free"], valid, collision)
    )
    head_distributions["ttc"]["within_frame_safety_ranking"] = (
        _within_frame_safety_ranking(scores["ttc"], valid, collision)
    )
    rescue = raw_collision & ~selected_collision
    introduced = ~raw_collision & selected_collision
    return {
        "n_frames": n,
        "thresholds": {
            "collision_free": collision_free_threshold,
            "ttc": ttc_threshold,
        },
        "candidate_collision_rate": float(collision[valid].mean()),
        "raw_collision_rate": float(raw_collision.mean()),
        "oracle_collision_rate": float((~safe_candidate).mean()),
        "selected_collision_rate": float(selected_collision.mean()),
        "rescue_rate": float(rescue.mean()),
        "introduced_collision_rate": float(introduced.mean()),
        "rescue_given_raw_collision_and_safe_alternative": float(
            rescue.sum() / max((raw_collision & safe_alternative).sum(), 1)
        ),
        "switch_rate": float((selected != 0).mean()),
        "fallback_rate": float((~has_feasible).mean()),
        "mean_feasible_candidates": float(feasible.sum(1).mean()),
        "raw_feasible_rate": float(feasible[:, 0].mean()),
        "raw_mae_mps": float(imitation_error[:, 0].mean()),
        "selected_mae_mps": float(imitation_error[rows, selected].mean()),
        "selected_candidate_index": _distribution(selected),
        "head_distributions": head_distributions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--scorer", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples-per-shard", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--collision-free-threshold", type=float, default=0.5)
    parser.add_argument("--ttc-threshold", type=float, default=0.5)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    if args.samples_per_shard < 0 or args.batch_size <= 0:
        parser.error("samples-per-shard must be nonnegative and batch-size positive")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from lead.tfv6.trajectory_scene_scorer import (
        SCORE_HEADS,
        load_trajectory_scene_scorer,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    arrays = _sample_features(Path(args.feature_cache_dir), args.samples_per_shard)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(arrays["candidate_states"]),
            torch.from_numpy(arrays["scene_tokens"]),
            torch.from_numpy(arrays["candidate_valid"]),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    result = {"device": str(device), "models": {}}
    for path in args.scorer:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        scorer = load_trajectory_scene_scorer(checkpoint, device).eval()
        outputs = {head: [] for head in SCORE_HEADS}
        with torch.inference_mode():
            for state, scene, valid in loader:
                logits = scorer(
                    state.to(device, dtype=torch.float32),
                    scene.to(device, dtype=torch.float32),
                    valid.to(device),
                )
                for head in SCORE_HEADS:
                    outputs[head].append(
                        torch.sigmoid(logits[head].float()).cpu().numpy()
                    )
        scores = {head: np.concatenate(parts) for head, parts in outputs.items()}
        audit = _audit_scores(
            arrays, scores, args.collision_free_threshold, args.ttc_threshold
        )
        audit["epoch"] = checkpoint["epoch"]
        result["models"][str(Path(path).resolve())] = audit
        print(
            f"epoch={audit['epoch']} n={audit['n_frames']} "
            f"collision={audit['raw_collision_rate']:.2%}"
            f"->{audit['selected_collision_rate']:.2%} "
            f"rescue={audit['rescue_rate']:.2%} "
            f"introduced={audit['introduced_collision_rate']:.2%} "
            f"switch={audit['switch_rate']:.1%} "
            f"fallback={audit['fallback_rate']:.1%}"
        )
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
