"""Held-out audit for B2d': learned future risk controls speed, never route."""

from __future__ import annotations

import argparse
import json
import lzma
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


@torch.inference_mode()
def _predict(head, arrays, device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    for start in range(0, len(arrays.keys), batch_size):
        end = min(start + batch_size, len(arrays.keys))
        features = torch.from_numpy(
            arrays.route_features[start:end].astype(np.float32),
        ).to(device)
        current = torch.from_numpy(
            arrays.current_speed[start:end].astype(np.float32),
        ).to(device)
        target = torch.from_numpy(
            arrays.target_speed[start:end].astype(np.float32),
        ).to(device)
        outputs.append(torch.sigmoid(head(features, current, target)).cpu().numpy())
    return np.concatenate(outputs)


def _load_expert_targets(manifest_path: str, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with open(manifest_path) as handle:
        entries = {
            entry["key"]: entry
            for line in handle
            if line.strip()
            for entry in (json.loads(line),)
        }
    missing = set(keys.tolist()) - set(entries)
    if missing:
        raise KeyError(f"{len(missing)} feature keys are absent from {manifest_path}")

    speed = np.empty(len(keys), dtype=np.float32)
    brake = np.empty(len(keys), dtype=bool)
    for index, key in enumerate(keys.tolist()):
        entry = entries[key]
        meta_path = Path(entry["src"]).parent.parent / "metas" / f"{entry['frame']}.pkl"
        with lzma.open(meta_path, "rb") as handle:
            meta = pickle.load(handle)
        brake[index] = bool(meta["brake"])
        speed[index] = 0.0 if brake[index] else float(meta["target_speed"])
    return speed, brake


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--gate-pairs",
        default=(
            "0.1:0.5,0.3:0.7,0.5:0.9,0.7:0.95,"
            "0.8:0.98,0.9:0.99,0.95:0.995"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from lead.tfv6.route_safety_head import (
        binary_average_precision,
        load_route_safety_head,
        load_safety_feature_dir,
    )
    from scripts.p4.eval_b2c_prime_speed_gate import _gate_pairs, _scope_metrics

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.head, map_location=device, weights_only=True)
    head = load_route_safety_head(checkpoint, device).eval().requires_grad_(False)
    arrays = load_safety_feature_dir(args.feature_cache_dir)
    risk = _predict(head, arrays, device, args.batch_size)

    count = len(arrays.keys) if args.limit is None else min(args.limit, len(arrays.keys))
    keys = arrays.keys[:count]
    expert_speed, expert_brake = _load_expert_targets(args.heldout_manifest, keys)
    valid = arrays.valid[:count]
    confidence = arrays.confidence[:count].astype(np.float32)
    selected = np.where(valid, confidence, -np.inf).argmax(axis=1)
    frame = np.arange(count)

    values = {
        "pred_risk": torch.from_numpy(risk[:count][frame, selected].astype(np.float32)),
        "gt_risk": torch.from_numpy(
            arrays.collision[:count][frame, selected].astype(np.float32),
        ),
        "raw_speed": torch.from_numpy(arrays.target_speed[:count].astype(np.float32)),
        "expert_speed": torch.from_numpy(expert_speed),
        "expert_brake": torch.from_numpy(expert_brake),
        "route_ade": torch.from_numpy(
            arrays.ade[:count][frame, selected].astype(np.float32),
        ),
        "multi": torch.from_numpy(valid.sum(axis=1) > 1),
    }
    pairs = _gate_pairs(args.gate_pairs)
    scopes = {}
    for name, mask in (
        ("all", torch.ones(count, dtype=torch.bool)),
        ("multi", values["multi"]),
    ):
        result = _scope_metrics(
            values["pred_risk"][mask],
            values["gt_risk"][mask],
            values["raw_speed"][mask],
            values["expert_speed"][mask],
            values["expert_brake"][mask],
            values["route_ade"][mask],
            pairs,
            0.5,
        )
        labels = values["gt_risk"][mask].numpy().astype(bool)
        scores = values["pred_risk"][mask].numpy()
        result["common"]["risk_average_precision"] = binary_average_precision(scores, labels)
        scopes[name] = result

    payload = {
        "head": args.head,
        "source_checkpoint": checkpoint.get("source_checkpoint"),
        "feature_cache_dir": args.feature_cache_dir,
        "heldout_manifest": args.heldout_manifest,
        "n_frames": count,
        "route_selection": "highest-confidence valid arm (never changed)",
        "label": "future dynamic collision on selected arm",
        "scopes": scopes,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nB2d' future-risk speed gate ({count} held-out frames; route is never changed)")
    for scope in ("all", "multi"):
        result = scopes[scope]
        common = result["common"]
        print(
            f"\n[{scope}] n={common['n']} ADE={common['route_ade']:.3f} "
            f"risk AUROC={common['risk_auroc']:.3f} "
            f"AP={common['risk_average_precision']:.3f} "
            f"future-collision={common['gt_unsafe_rate'] * 100:.1f}%"
        )
        print(
            f"  expert brake={common['expert_brake_rate'] * 100:.1f}% "
            f"model brake={common['model_brake_rate'] * 100:.1f}% "
            f"missed expert brake={common['missed_expert_brake_rate'] * 100:.1f}% "
            f"speed MAE={common['raw_target_speed_mae']:.3f} m/s"
        )
        print(
            " low high | slow stop | collision recall/prec/false | "
            "brake slow/stop rescue false-go slow/stop | speed raw->gated  MAE"
        )
        for low, high in pairs:
            gate = result["gates"][f"{low:g},{high:g}"]
            print(
                f" {low:.2f} {high:.2f} | "
                f"{gate['slow_rate'] * 100:4.1f}% {gate['stop_rate'] * 100:4.1f}% | "
                f"{gate['unsafe_recall'] * 100:4.1f}/"
                f"{gate['slow_precision'] * 100:4.1f}/"
                f"{gate['false_slow_rate_on_safe'] * 100:4.1f}% | "
                f"{gate['missed_expert_brake_slow_recovery'] * 100:4.1f}/"
                f"{gate['missed_expert_brake_stop_recovery'] * 100:4.1f}% "
                f"{gate['false_slow_rate_on_expert_go'] * 100:4.1f}/"
                f"{gate['false_stop_rate_on_expert_go'] * 100:4.1f}% | "
                f"{common['raw_target_speed']:.2f}->{gate['mean_gated_speed']:.2f} "
                f"MAE={gate['target_speed_mae']:.3f}"
            )
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
