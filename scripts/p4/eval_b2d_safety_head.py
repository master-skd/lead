"""Evaluate B2d risk calibration and confidence-preserving gated route selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


@torch.inference_mode()
def _predict(head, arrays, device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    for start in range(0, len(arrays.keys), batch_size):
        end = min(start + batch_size, len(arrays.keys))
        features = torch.from_numpy(arrays.route_features[start:end].astype(np.float32)).to(device)
        current = torch.from_numpy(arrays.current_speed[start:end].astype(np.float32)).to(device)
        target = torch.from_numpy(arrays.target_speed[start:end].astype(np.float32)).to(device)
        outputs.append(torch.sigmoid(head(features, current, target)).cpu().numpy())
    return np.concatenate(outputs)


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator / max(denominator, 1))


def evaluate_gate(arrays, risk: np.ndarray, unsafe_threshold: float, safe_threshold: float, mask=None) -> dict:
    if mask is None:
        mask = np.ones(len(arrays.keys), dtype=bool)
    valid = arrays.valid[mask]
    confidence = arrays.confidence[mask].astype(np.float32)
    collision = arrays.collision[mask]
    ade = arrays.ade[mask].astype(np.float32)
    risk = risk[mask]
    n, k = valid.shape
    ar = np.arange(n)
    base = np.where(valid, confidence, -np.inf).argmax(axis=1)
    base_collision = collision[ar, base]
    trigger = risk[ar, base] >= unsafe_threshold
    candidate = valid & (risk < safe_threshold)
    candidate[:, :] &= np.arange(k)[None, :] != base[:, None]
    has_alternate = candidate.any(axis=1)
    alternate = np.where(candidate, confidence, -np.inf).argmax(axis=1)
    switched = trigger & has_alternate
    selected = np.where(switched, alternate, base)
    selected_collision = collision[ar, selected]
    truly_safe_alternate = (valid & ~collision) & (np.arange(k)[None, :] != base[:, None])
    rescue_possible = base_collision & truly_safe_alternate.any(axis=1)
    rescued = base_collision & ~selected_collision
    false_switch = switched & ~base_collision
    introduced_collision = ~base_collision & selected_collision
    fallback_gate = trigger & ~has_alternate

    return {
        "n": n,
        "unsafe_threshold": unsafe_threshold,
        "safe_threshold": safe_threshold,
        "trigger_rate": float(trigger.mean()),
        "switch_rate": float(switched.mean()),
        "fallback_speed_gate_rate": float(fallback_gate.mean()),
        "collision_before": float(base_collision.mean()),
        "collision_after": float(selected_collision.mean()),
        "rescue_possible_count": int(rescue_possible.sum()),
        "rescued_count": int((rescued & rescue_possible).sum()),
        "rescue_recall": _safe_div(int((rescued & rescue_possible).sum()), int(rescue_possible.sum())),
        "false_switch_rate": float(false_switch.mean()),
        "introduced_collision_rate": float(introduced_collision.mean()),
        "ade_before": float(ade[ar, base].mean()),
        "ade_after": float(ade[ar, selected].mean()),
        "ade_delta": float((ade[ar, selected] - ade[ar, base]).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-false-switch-rate", type=float, default=0.005)
    parser.add_argument("--max-ade-delta", type=float, default=0.02)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from lead.tfv6.route_safety_head import (
        binary_average_precision,
        binary_auroc,
        load_route_safety_head,
        load_safety_feature_dir,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.head, map_location=device, weights_only=True)
    head = load_route_safety_head(checkpoint, device).eval()
    arrays = load_safety_feature_dir(args.feature_cache_dir)
    risk = _predict(head, arrays, device, args.batch_size)
    arm_label = arrays.collision[arrays.valid]
    arm_score = risk[arrays.valid]
    masked_conf = np.where(arrays.valid, arrays.confidence, -np.inf)
    base = masked_conf.argmax(axis=1)
    ar = np.arange(len(base))
    base_label = arrays.collision[ar, base]
    base_score = risk[ar, base]
    arm_metrics = {
        "n_valid_arms": int(arrays.valid.sum()),
        "collision_rate": float(arm_label.mean()),
        "auroc": binary_auroc(arm_score, arm_label),
        "average_precision": binary_average_precision(arm_score, arm_label),
        "base_auroc": binary_auroc(base_score, base_label),
        "base_average_precision": binary_average_precision(base_score, base_label),
    }

    grid = []
    for high in (0.3, 0.5, 0.7, 0.8, 0.9):
        for low in (0.1, 0.2, 0.3, 0.5):
            if low >= high:
                continue
            all_metrics = evaluate_gate(arrays, risk, high, low)
            multi = arrays.valid.sum(axis=1) > 1
            row = {"all": all_metrics}
            if multi.any():
                row["multi"] = evaluate_gate(arrays, risk, high, low, multi)
            grid.append(row)
    feasible = [
        row for row in grid
        if row["all"]["false_switch_rate"] <= args.max_false_switch_rate
        and row["all"]["ade_delta"] <= args.max_ade_delta
    ]
    recommended = min(
        feasible,
        key=lambda row: (
            row["all"]["collision_after"],
            -row["all"]["rescued_count"],
            row["all"]["switch_rate"],
        ),
    ) if feasible else None
    payload = {
        "head": args.head,
        "source_checkpoint": checkpoint.get("source_checkpoint"),
        "feature_cache_dir": args.feature_cache_dir,
        "n_frames": len(arrays.keys),
        "arm_metrics": arm_metrics,
        "constraints": {
            "max_false_switch_rate": args.max_false_switch_rate,
            "max_ade_delta": args.max_ade_delta,
        },
        "recommended": recommended,
        "threshold_grid": grid,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(
        f"B2d safety head: frames={len(arrays.keys)} arm collision={arm_metrics['collision_rate']:.2%} "
        f"AUROC={arm_metrics['auroc']:.4f} AP={arm_metrics['average_precision']:.4f}"
    )
    if recommended is None:
        print("no threshold pair satisfies the false-switch/ADE constraints")
    else:
        chosen = recommended["all"]
        print(
            f"recommended low/high={chosen['safe_threshold']:.2f}/{chosen['unsafe_threshold']:.2f} "
            f"collision {chosen['collision_before']:.2%}->{chosen['collision_after']:.2%} "
            f"switch={chosen['switch_rate']:.2%} false={chosen['false_switch_rate']:.2%} "
            f"ADE delta={chosen['ade_delta']:.4f}"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
