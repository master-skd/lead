"""Evaluate learned velocity preference/risk and confidence-preserving selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


@torch.inference_mode()
def predict(scorer, arrays, device: torch.device, batch_size: int):
    preferences, risks = [], []
    for start in range(0, len(arrays.keys), batch_size):
        end = min(start + batch_size, len(arrays.keys))
        route = torch.from_numpy(
            arrays.route_features[start:end].astype(np.float32)
        ).to(device)
        current = torch.from_numpy(
            arrays.current_speed[start:end].astype(np.float32)
        ).to(device)
        raw = torch.from_numpy(
            arrays.raw_target_speed[start:end].astype(np.float32)
        ).to(device)
        velocity = torch.from_numpy(
            arrays.candidate_velocity[start:end].astype(np.float32)
        ).to(device)
        preference, risk = scorer(route, current, raw, velocity)
        preferences.append(preference.cpu().numpy())
        risks.append(torch.sigmoid(risk).cpu().numpy())
    return np.concatenate(preferences), np.concatenate(risks)


def evaluate_gate(
    arrays,
    preference: np.ndarray,
    risk: np.ndarray,
    unsafe_threshold: float,
    safe_threshold: float,
    mask: np.ndarray | None = None,
) -> dict:
    if mask is None:
        mask = np.ones(len(arrays.keys), dtype=bool)
    valid = arrays.candidate_valid[mask]
    collision = arrays.collision[mask]
    error = arrays.imitation_error[mask].astype(np.float32)
    preference = preference[mask]
    risk = risk[mask]
    n, candidates = valid.shape
    rows = np.arange(n)
    raw_collision = collision[:, 0]
    trigger = risk[:, 0] >= unsafe_threshold
    alternate = valid & (risk < safe_threshold)
    alternate[:, 0] = False
    has_alternate = alternate.any(axis=1)
    alternate_index = np.where(alternate, preference, -np.inf).argmax(axis=1)
    switched = trigger & has_alternate
    selected = np.where(switched, alternate_index, 0)
    selected_collision = collision[rows, selected]
    safe_alternate = valid & ~collision
    safe_alternate[:, 0] = False
    rescue_possible = raw_collision & safe_alternate.any(axis=1)
    rescued = raw_collision & ~selected_collision
    false_switch = switched & ~raw_collision
    introduced_collision = ~raw_collision & selected_collision
    return {
        "n": n,
        "candidate_count": candidates,
        "unsafe_threshold": unsafe_threshold,
        "safe_threshold": safe_threshold,
        "trigger_rate": float(trigger.mean()),
        "switch_rate": float(switched.mean()),
        "fallback_rate": float((trigger & ~has_alternate).mean()),
        "collision_before": float(raw_collision.mean()),
        "collision_after": float(selected_collision.mean()),
        "rescue_possible_count": int(rescue_possible.sum()),
        "rescued_count": int((rescued & rescue_possible).sum()),
        "rescue_recall": _safe_div(
            int((rescued & rescue_possible).sum()), int(rescue_possible.sum())
        ),
        "false_switch_rate": float(false_switch.mean()),
        "introduced_collision_rate": float(introduced_collision.mean()),
        "profile_mae_before": float(error[:, 0].mean()),
        "profile_mae_after": float(error[rows, selected].mean()),
        "profile_mae_delta": float((error[rows, selected] - error[:, 0]).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-false-switch-rate", type=float, default=0.01)
    parser.add_argument("--max-profile-mae-delta", type=float, default=0.1)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from lead.tfv6.route_safety_head import binary_auroc, binary_average_precision
    from lead.tfv6.velocity_scorer import (
        load_velocity_feature_dir,
        load_velocity_scorer,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.scorer, map_location=device, weights_only=True)
    scorer = load_velocity_scorer(checkpoint, device).eval().requires_grad_(False)
    arrays = load_velocity_feature_dir(args.feature_cache_dir)
    preference, risk = predict(scorer, arrays, device, args.batch_size)
    valid = arrays.candidate_valid
    rows = np.arange(len(arrays.keys))
    imitation_selected = np.where(valid, preference, -np.inf).argmax(axis=1)
    risk_labels = arrays.collision[valid]
    risk_scores = risk[valid]
    metrics = {
        "n_frames": len(arrays.keys),
        "n_valid_candidates": int(valid.sum()),
        "collision_rate": float(risk_labels.mean()),
        "collision_auroc": binary_auroc(risk_scores, risk_labels),
        "collision_average_precision": binary_average_precision(
            risk_scores, risk_labels
        ),
        "imitation_top1_accuracy": float(
            (imitation_selected == arrays.imitation_target).mean()
        ),
        "raw_profile_mae_mps": float(arrays.imitation_error[:, 0].mean()),
        "selected_profile_mae_mps": float(
            arrays.imitation_error[rows, imitation_selected].mean()
        ),
        "oracle_profile_mae_mps": float(
            arrays.imitation_error[rows, arrays.imitation_target].mean()
        ),
    }

    grid = []
    for high in (0.3, 0.5, 0.7, 0.8, 0.9):
        for low in (0.1, 0.2, 0.3, 0.5):
            if low >= high:
                continue
            all_metrics = evaluate_gate(arrays, preference, risk, high, low)
            row = {"all": all_metrics}
            if arrays.multi.any():
                row["multi"] = evaluate_gate(
                    arrays, preference, risk, high, low, arrays.multi
                )
            grid.append(row)
    feasible = [
        row
        for row in grid
        if row["all"]["false_switch_rate"] <= args.max_false_switch_rate
        and row["all"]["profile_mae_delta"] <= args.max_profile_mae_delta
    ]
    recommended = (
        min(
            feasible,
            key=lambda row: (
                row["all"]["collision_after"],
                -row["all"]["rescued_count"],
                row["all"]["introduced_collision_rate"],
                row["all"]["switch_rate"],
            ),
        )
        if feasible
        else None
    )
    payload = {
        "scorer": str(Path(args.scorer).resolve()),
        "feature_cache_dir": str(Path(args.feature_cache_dir).resolve()),
        "source_checkpoint": checkpoint.get("source_checkpoint"),
        "source_vocabulary": checkpoint.get("source_vocabulary"),
        "metrics": metrics,
        "constraints": {
            "max_false_switch_rate": args.max_false_switch_rate,
            "max_profile_mae_delta": args.max_profile_mae_delta,
        },
        "recommended": recommended,
        "threshold_grid": grid,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(
        f"B3a velocity scorer: frames={len(arrays.keys)} "
        f"profile MAE raw/model/oracle={metrics['raw_profile_mae_mps']:.3f}/"
        f"{metrics['selected_profile_mae_mps']:.3f}/"
        f"{metrics['oracle_profile_mae_mps']:.3f} m/s "
        f"risk AUROC={metrics['collision_auroc']:.4f} "
        f"AP={metrics['collision_average_precision']:.4f}"
    )
    if recommended is None:
        print("no threshold pair satisfies false-switch/profile-MAE constraints")
    else:
        chosen = recommended["all"]
        print(
            f"recommended low/high={chosen['safe_threshold']:.2f}/"
            f"{chosen['unsafe_threshold']:.2f} collision "
            f"{chosen['collision_before']:.2%}->{chosen['collision_after']:.2%} "
            f"rescue={chosen['rescue_recall']:.1%} switch={chosen['switch_rate']:.2%} "
            f"false={chosen['false_switch_rate']:.2%} "
            f"MAE delta={chosen['profile_mae_delta']:.3f}"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
