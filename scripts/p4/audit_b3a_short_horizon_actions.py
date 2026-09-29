"""Final no-training coverage test for PID-executable slowdown actions.

Reuse the existing control-aware GT collision labels, but measure progress and
cached-Path coverage at 1.0 and 1.5 seconds. No model forward or GPU is needed.
This is an oracle upper bound, not a deployable risk predictor or CARLA replay.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lead.data_loader.future_actor_cache import FUTURE_TIMES_S
from lead.expert.config_expert import ExpertConfig
from lead.tfv6.control_aware_velocity_oracle import (
    ACTION_NAMES,
    Dynamics,
    rollout_scalar_target,
)


def _wilson_interval(success: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.96
    p = success / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    half /= denominator
    return [float(center - half), float(center + half)]


def _route_ends_and_check_targets(
    feature_dir: Path, keys: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    requested = {str(key): index for index, key in enumerate(keys)}
    if len(requested) != len(keys):
        raise ValueError("duplicate keys in control-aware oracle labels")
    route_ends = np.full(len(keys), np.nan, dtype=np.float32)
    seen = 0
    for path in sorted(feature_dir.glob("velocity_features_*.npz")):
        with np.load(path, allow_pickle=False) as file:
            shard_keys = file["keys"]
            velocity = file["candidate_velocity"].astype(np.float32)
            valid = file["candidate_valid"]
            raw_target = file["raw_target_speed"].astype(np.float32)
        if velocity.shape[2] != len(FUTURE_TIMES_S):
            raise ValueError(f"unexpected candidate horizon in {path}")
        last_distance = velocity.sum(axis=2) * float(FUTURE_TIMES_S[0])
        last_distance[~valid] = -np.inf
        ends = last_distance.max(axis=1)
        for row, key in enumerate(shard_keys):
            index = requested.get(str(key))
            if index is None:
                continue
            if np.isfinite(route_ends[index]):
                raise ValueError(f"duplicate selected key across feature shards: {key}")
            if not np.isclose(raw_target[row], targets[index, 0], atol=0.03):
                raise ValueError(f"raw target differs from oracle labels for {key}")
            route_ends[index] = ends[row]
            seen += 1
    if seen != len(keys) or not np.isfinite(route_ends).all():
        raise ValueError(f"matched {seen}/{len(keys)} oracle frames to feature cache")
    return route_ends


def audit_horizon(
    *,
    horizon_s: float,
    keys: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
    collisions_2s: np.ndarray,
    ttc_s: np.ndarray,
    current_speeds: np.ndarray,
    route_ends: np.ndarray,
    dynamics: Dynamics,
    max_progress_loss_m: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    steps = int(round(horizon_s / float(FUTURE_TIMES_S[0])))
    if steps < 1 or steps > len(FUTURE_TIMES_S):
        raise ValueError("horizon must be within the cached future interval")
    times = FUTURE_TIMES_S[:steps]
    progress = np.full(targets.shape, np.nan, dtype=np.float32)
    terminal_speed = np.full(targets.shape, np.nan, dtype=np.float32)
    config = ExpertConfig()
    for frame in range(len(keys)):
        for action in np.flatnonzero(valid[frame]):
            rollout = rollout_scalar_target(
                float(current_speeds[frame]),
                float(targets[frame, action]),
                dynamics=dynamics,
                expert_config=config,
                times_s=times,
            )
            progress[frame, action] = rollout.progress_m
            terminal_speed[frame, action] = float(rollout.speed_mps[-1])
    unsafe = collisions_2s & (ttc_s <= horizon_s + 1e-5)
    raw_qualified = progress[:, 0] <= route_ends + 0.01
    geometry_valid = progress <= route_ends[:, None] + 0.01
    slow = targets <= targets[:, :1] + 1e-3
    enough_progress = progress >= progress[:, :1] - max_progress_loss_m
    safe = valid & ~unsafe & geometry_valid & slow & enough_progress
    moving = safe & (targets > 0.01)
    raw_unsafe_qualified = unsafe[:, 0] & raw_qualified
    rescued_any = raw_unsafe_qualified & safe.any(axis=1)
    rescued_moving = raw_unsafe_qualified & moving.any(axis=1)
    rescued_stop_only = rescued_any & ~rescued_moving
    rescue_actions = Counter()
    progress_loss = []
    for frame in np.flatnonzero(rescued_moving):
        candidates = np.flatnonzero(moving[frame])
        selected = int(candidates[np.argmax(progress[frame, candidates])])
        rescue_actions[ACTION_NAMES[selected]] += 1
        progress_loss.append(float(progress[frame, 0] - progress[frame, selected]))
    raw_count = int(raw_unsafe_qualified.sum())
    moving_count = int(rescued_moving.sum())
    unique_rescue_routes = len(
        {str(keys[i]).rsplit("__", 1)[0] for i in np.flatnonzero(rescued_moving)}
    )
    summary = {
        "horizon_s": horizon_s,
        "frames": len(keys),
        "geometry_qualified_frames": int(raw_qualified.sum()),
        "raw_unsafe_all_frames": int(unsafe[:, 0].sum()),
        "raw_unsafe_qualified_frames": raw_count,
        "moving_rescue_frames": moving_count,
        "stop_only_rescue_frames": int(rescued_stop_only.sum()),
        "moving_rescue_given_unsafe": moving_count / raw_count if raw_count else None,
        "moving_rescue_rate_wilson95": _wilson_interval(moving_count, raw_count),
        "moving_rescue_routes": unique_rescue_routes,
        "moving_rescue_action_counts": dict(rescue_actions),
        "moving_rescue_progress_loss_mean_m": (
            float(np.mean(progress_loss)) if progress_loss else None
        ),
        "coverage_gate": (
            "inconclusive"
            if raw_count < 50
            else "pass"
            if moving_count / raw_count >= 0.20
            else "fail"
        ),
    }
    details = {
        "progress_m": progress,
        "terminal_speed_mps": terminal_speed,
        "raw_geometry_qualified": raw_qualified,
        "raw_unsafe": unsafe[:, 0],
        "moving_rescue": rescued_moving,
        "stop_only_rescue": rescued_stop_only,
    }
    return summary, details


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-report",
        type=Path,
        default=root
        / "outputs/local_training/p5_stepB3a_control_aware_oracle/nominal_5000.json",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=root
        / "outputs/local_training/p5_stepB3a_v2_scene_scorer/features/heldout",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "outputs/local_training/p5_stepB3a_control_aware_oracle/short_horizon_5000.json",
    )
    parser.add_argument("--max-progress-loss-m", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_progress_loss_m < 0:
        parser.error("--max-progress-loss-m must be nonnegative")
    report = json.loads(args.oracle_report.read_text())
    if report.get("selected_paths") is not None:
        raise ValueError("this audit requires the original cached-feature oracle")
    if Path(report["features"]).resolve() != args.features.resolve():
        raise ValueError("feature directory differs from source oracle report")
    dynamics = Dynamics(**report["dynamics"])
    with np.load(report["frame_labels"], allow_pickle=False) as file:
        arrays = {
            name: file[name]
            for name in (
                "keys",
                "targets_mps",
                "valid",
                "collision",
                "ttc_s",
                "progress_m",
            )
        }
    keys = arrays["keys"]
    targets = arrays["targets_mps"].astype(np.float32)
    valid = arrays["valid"].astype(bool)
    collision = arrays["collision"].astype(bool)
    ttc = arrays["ttc_s"].astype(np.float32)
    route_ends = _route_ends_and_check_targets(args.features, keys, targets)
    current_speeds = np.zeros(len(keys), dtype=np.float32)
    # The source oracle did not store current speed. Read it once from the same
    # feature shards, by key, to avoid any new model forward or approximation.
    indices = {str(key): index for index, key in enumerate(keys)}
    for path in sorted(args.features.glob("velocity_features_*.npz")):
        with np.load(path, allow_pickle=False) as file:
            for key, speed in zip(file["keys"], file["current_speed"], strict=True):
                index = indices.get(str(key))
                if index is not None:
                    current_speeds[index] = float(speed)
    if np.any(current_speeds < 0):
        raise ValueError("negative current speed in feature cache")
    results = {}
    frame_details = {"keys": keys, "route_end_m": route_ends}
    for horizon in (1.0, 1.5, 2.0):
        summary, details = audit_horizon(
            horizon_s=horizon,
            keys=keys,
            targets=targets,
            valid=valid,
            collisions_2s=collision,
            ttc_s=ttc,
            current_speeds=current_speeds,
            route_ends=route_ends,
            dynamics=dynamics,
            max_progress_loss_m=args.max_progress_loss_m,
        )
        if horizon == 2.0 and not np.allclose(
            details["progress_m"][valid], arrays["progress_m"][valid], atol=0.03
        ):
            raise ValueError("2 s rollout does not reproduce source oracle progress")
        results[str(horizon)] = summary
        for name, value in details.items():
            frame_details[f"{name}_{horizon}"] = value
    output = {
        "scope": "fixed selected Path reconstructed from existing feature states; PID scalar target actions; GT actor collision labels reused from source oracle",
        "source_oracle_report": str(args.oracle_report.resolve()),
        "features": str(args.features.resolve()),
        "frames": len(keys),
        "max_progress_loss_m": args.max_progress_loss_m,
        "coverage_gate_rule": "pass if at least 50 qualified raw-unsafe frames and >=20% have a moving, slowdown-only, progress-feasible rescue; else fail or inconclusive",
        "horizons": results,
        "limitations": "GT future actors, reconstructed Path, approximate vehicle dynamics, fixed scalar target and fixed Path; no predicted-risk observability or closed-loop validation. Passing coverage is necessary, not sufficient, for a deployable velocity scorer.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_path = args.output.with_suffix(".frames.npz")
    np.savez_compressed(frame_path, **frame_details)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    print(f"wrote {args.output} and {frame_path}")


if __name__ == "__main__":
    main()
