"""Held-out threshold sweep for the B3a-v2 trajectory-scene scorer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _thresholds(value: str) -> list[float]:
    result = [float(item) for item in value.split(",")]
    if not result or any(not 0.0 < item < 1.0 for item in result):
        raise argparse.ArgumentTypeError(
            "thresholds must be comma-separated values in (0,1)"
        )
    return result


def _metrics(
    scores: dict[str, np.ndarray],
    valid: np.ndarray,
    collision: np.ndarray,
    imitation_error: np.ndarray,
    collision_free_threshold: float,
    ttc_threshold: float,
) -> dict[str, float | int]:
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
    preferred = np.where(feasible, utility, -np.inf).argmax(axis=1)
    safety = 0.7 * scores["collision_free"] + 0.3 * scores["ttc"]
    fallback_index = np.where(valid, safety, -np.inf).argmax(axis=1)
    has_feasible = feasible.any(axis=1)
    selected = np.where(has_feasible, preferred, fallback_index)
    rows = np.arange(len(selected))
    raw_collision = collision[:, 0]
    selected_collision = collision[rows, selected]
    return {
        "collision_free_threshold": collision_free_threshold,
        "ttc_threshold": ttc_threshold,
        "switch_rate": float((selected != 0).mean()),
        "fallback_rate": float((~has_feasible).mean()),
        "raw_collision_rate": float(raw_collision.mean()),
        "selected_collision_rate": float(selected_collision.mean()),
        "rescue_rate": float((raw_collision & ~selected_collision).mean()),
        "introduced_collision_rate": float(
            (~raw_collision & selected_collision).mean()
        ),
        "selected_imitation_mae_mps": float(imitation_error[rows, selected].mean()),
        "selected_progress": float(scores["progress"][rows, selected].mean()),
        "selected_comfort": float(scores["comfort"][rows, selected].mean()),
        "selected_imitation_score": float(scores["imitation"][rows, selected].mean()),
        "mean_feasible_candidates": float(feasible.sum(axis=1).mean()),
        "n_frames": len(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--collision-free-thresholds", type=_thresholds, default=[0.3, 0.5, 0.7]
    )
    parser.add_argument("--ttc-thresholds", type=_thresholds, default=[0.3, 0.5, 0.7])
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from lead.tfv6.route_safety_head import binary_auroc, binary_average_precision
    from lead.tfv6.trajectory_scene_scorer import (
        SCORE_HEADS,
        load_trajectory_scene_feature_dir,
        load_trajectory_scene_scorer,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arrays = load_trajectory_scene_feature_dir(args.feature_cache_dir)
    checkpoint = torch.load(args.scorer, map_location=device, weights_only=True)
    scorer = (
        load_trajectory_scene_scorer(checkpoint, device).eval().requires_grad_(False)
    )
    dataset = TensorDataset(
        torch.from_numpy(arrays.scene_tokens),
        torch.from_numpy(arrays.candidate_states),
        torch.from_numpy(arrays.candidate_valid),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    output = {name: [] for name in SCORE_HEADS}
    with torch.inference_mode():
        for scene, state, valid in loader:
            logits = scorer(
                state.to(device, dtype=torch.float32, non_blocking=True),
                scene.to(device, dtype=torch.float32, non_blocking=True),
                valid.to(device, dtype=torch.bool, non_blocking=True),
            )
            for name in SCORE_HEADS:
                output[name].append(torch.sigmoid(logits[name].float()).cpu().numpy())
    scores = {name: np.concatenate(parts) for name, parts in output.items()}
    valid = arrays.candidate_valid.astype(bool)
    collision = arrays.collision.astype(bool)
    risk = 1.0 - scores["collision_free"][valid]
    labels = collision[valid]
    rows = []
    for collision_free_threshold in args.collision_free_thresholds:
        for ttc_threshold in args.ttc_thresholds:
            row = _metrics(
                scores,
                valid,
                collision,
                arrays.imitation_error,
                collision_free_threshold,
                ttc_threshold,
            )
            rows.append(row)
            print(
                f"cf={collision_free_threshold:.2f} ttc={ttc_threshold:.2f} "
                f"switch={row['switch_rate']:.1%} fallback={row['fallback_rate']:.1%} "
                f"collision={row['raw_collision_rate']:.2%}->{row['selected_collision_rate']:.2%} "
                f"rescue={row['rescue_rate']:.2%} introduced={row['introduced_collision_rate']:.2%} "
                f"MAE={row['selected_imitation_mae_mps']:.3f}"
            )
    result = {
        "scorer": str(Path(args.scorer).resolve()),
        "feature_cache": str(Path(args.feature_cache_dir).resolve()),
        "collision_auroc": binary_auroc(risk, labels),
        "collision_average_precision": binary_average_precision(risk, labels),
        "candidate_collision_rate": float(labels.mean()),
        "threshold_sweep": rows,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
