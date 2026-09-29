"""Held-out, fixed-Path audit of the baseline-preserving velocity guard.

GT actor futures provide collision labels. The longitudinal motion is a
rate-limited target-speed proxy, not a CARLA closed-loop rollout. This script
must qualify a scorer before the new mode is used for closed-loop evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lead.data_loader.future_actor_cache import unpack_future_actor_frame
from lead.tfv6.future_collision import future_collision_label
from lead.tfv6.velocity_scorer import (
    load_velocity_scorer,
    select_baseline_preserving_velocity,
)
from scripts.p4.audit_b3a_control_velocity_oracle import (
    actor_locations,
    interpolate_samples,
    reconstruct_route_samples,
)


def rate_limited_distances(
    current_speed: float,
    target_speed: float,
    *,
    max_accel: float = 1.89,
    max_decel: float = 4.95,
    interval_s: float = 0.25,
    steps: int = 8,
) -> np.ndarray:
    """Integrate a held target with bounded acceleration (control proxy)."""
    speed = max(0.0, float(current_speed))
    target = max(0.0, float(target_speed))
    distance = 0.0
    result = np.empty(steps, dtype=np.float32)
    for index in range(steps):
        next_speed = max(
            0.0,
            speed + np.clip(target - speed, -max_decel * interval_s, max_accel * interval_s),
        )
        distance += 0.5 * (speed + next_speed) * interval_s
        result[index] = distance
        speed = next_speed
    return result


def collision_under_target(samples, current, target, actors, margin):
    xy, yaw = interpolate_samples(samples, rate_limited_distances(current, target))
    return future_collision_label(
        xy, yaw, actors, safety_margin_m=margin, include_class_ids=(1, 2)
    ).collision


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=root / "outputs/local_training/p5_stepB3a_v2_scene_scorer/features/heldout")
    parser.add_argument("--future-actors", type=Path, default=root / "outputs/local_training/p5_stepB3a_v2_dense_data/future_actor_cache/heldout_future_actors")
    parser.add_argument("--scorer", type=Path, default=root / "outputs/local_training/p5_stepB3a_velocity_scorer/velocity_scorer_best.pth")
    parser.add_argument("--vocabulary", type=Path, default=root / "outputs/local_training/p5_stepB3a_relative_velocity_vocab/relative_velocity_vocab_k64.npy")
    parser.add_argument("--output", type=Path, default=root / "outputs/local_training/p5_stepB3a_baseline_guard/baseline_guard_heldout.json")
    parser.add_argument("--sample-per-shard", type=int, default=600)
    parser.add_argument("--limit", type=int, default=5000, help="0 evaluates all sampled frames")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--unsafe-threshold", type=float, default=0.9)
    parser.add_argument("--safe-threshold", type=float, default=0.3)
    parser.add_argument("--max-slowdown", type=float, default=1.0)
    parser.add_argument("--safety-margin", type=float, default=0.2)
    args = parser.parse_args()
    if args.sample_per_shard < 1 or args.batch_size < 1:
        raise ValueError("sample-per-shard and batch-size must be positive")
    paths = sorted(args.features.glob("velocity_features_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no feature shards in {args.features}")
    locations = actor_locations(args.future_actors)
    checkpoint = torch.load(args.scorer, map_location="cpu", weights_only=True)
    scorer = load_velocity_scorer(checkpoint, torch.device("cpu")).eval()
    vocabulary = torch.from_numpy(np.load(args.vocabulary).astype(np.float32))
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    loaded: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
    counts = dict(
        frames=0, triggered=0, switched=0, fallback=0,
        raw_collision=0, selected_collision=0, rescued=0, introduced=0,
        false_slow=0, raw_brake=0, raw_brake_overridden=0,
    )
    changes: list[float] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as cache:
            required = (
                "keys", "route_features", "current_speed", "raw_target_speed",
                "candidate_velocity", "candidate_valid", "candidate_states",
            )
            missing = [name for name in required if name not in cache]
            if missing:
                raise ValueError(f"{path} missing: {missing}")
            arrays = {name: cache[name] for name in required}
        n = len(arrays["keys"])
        rows = np.unique(np.linspace(0, n - 1, min(args.sample_per_shard, n), dtype=int))
        if args.limit:
            rows = rows[:max(0, args.limit - counts["frames"])]
        for start in tqdm(range(0, len(rows), args.batch_size), desc=path.stem, leave=False):
            indices = rows[start:start + args.batch_size]
            with torch.inference_mode():
                decision = select_baseline_preserving_velocity(
                    scorer,
                    torch.from_numpy(arrays["route_features"][indices].astype(np.float32)),
                    torch.from_numpy(arrays["current_speed"][indices].astype(np.float32)),
                    torch.from_numpy(arrays["raw_target_speed"][indices].astype(np.float32)),
                    vocabulary,
                    unsafe_threshold=args.unsafe_threshold,
                    safe_threshold=args.safe_threshold,
                    max_slowdown_mps=args.max_slowdown,
                )
            targets = decision.target_speed.reshape(-1).numpy()
            switched = decision.switched.numpy()
            triggered = decision.triggered.numpy()
            fallback = decision.fallback.numpy()
            for local, row in enumerate(indices):
                key = str(arrays["keys"][row])
                if key not in locations:
                    raise KeyError(f"missing actors for {key}")
                actor_path, actor_index = locations[key]
                if actor_path not in loaded:
                    with np.load(actor_path, allow_pickle=False) as actor_cache:
                        loaded[actor_path] = {name: actor_cache[name] for name in actor_cache.files}
                    if len(loaded) > 2:
                        loaded.popitem(last=False)
                actors = unpack_future_actor_frame(loaded[actor_path], actor_index)
                current = float(arrays["current_speed"][row])
                raw = float(arrays["raw_target_speed"][row])
                target = float(targets[local])
                samples = reconstruct_route_samples(
                    arrays["candidate_velocity"][row].astype(np.float32),
                    arrays["candidate_states"][row],
                    arrays["candidate_valid"][row].astype(bool),
                )
                raw_collision = collision_under_target(
                    samples, current, raw, actors, args.safety_margin
                )
                selected_collision = (
                    collision_under_target(
                        samples, current, target, actors, args.safety_margin
                    ) if switched[local] else raw_collision
                )
                counts["frames"] += 1
                counts["triggered"] += bool(triggered[local])
                counts["switched"] += bool(switched[local])
                counts["fallback"] += bool(fallback[local])
                counts["raw_collision"] += raw_collision
                counts["selected_collision"] += selected_collision
                counts["rescued"] += raw_collision and not selected_collision
                counts["introduced"] += not raw_collision and selected_collision
                counts["false_slow"] += bool(switched[local]) and not raw_collision
                counts["raw_brake"] += raw < 0.1
                counts["raw_brake_overridden"] += raw < 0.1 and abs(target - raw) > 1e-5
                if switched[local]:
                    changes.append(raw - target)
        if args.limit and counts["frames"] >= args.limit:
            break
    total = counts["frames"]
    report = {
        "scope": "fixed confidence Path, GT future actors, 2 s rate-limited target-speed proxy",
        "closed_loop_qualified": False,
        "next_step": "shadow only; compare would-select decisions with baseline route events before active evaluation",
        "scorer": str(args.scorer.resolve()),
        "vocabulary": str(args.vocabulary.resolve()),
        "thresholds": {
            "unsafe": args.unsafe_threshold,
            "safe": args.safe_threshold,
            "max_slowdown_mps": args.max_slowdown,
        },
        "counts": counts,
        "rates": {key: value / total for key, value in counts.items() if key != "frames"},
        "median_slowdown_when_switched_mps": float(np.median(changes)) if changes else 0.0,
        "limitations": "No actual CARLA controller/replanning, no predicted actor uncertainty; GT-actor proxy is not a closed-loop score.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
