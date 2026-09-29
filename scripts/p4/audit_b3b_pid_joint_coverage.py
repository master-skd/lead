"""GT upper bound for joint Path x PID-target actions on frozen B3b frames.

The longitudinal command uses LEAD's throttle/brake controller. Lateral motion
still uses the bounded-yaw proxy, so this is not a CARLA replay or a policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lead.data_loader.future_actor_cache import FUTURE_TIMES_S
from lead.expert.config_expert import ExpertConfig
from lead.tfv6.control_aware_velocity_oracle import (
    ACTION_NAMES, Dynamics, rollout_scalar_target, scalar_actions,
)
from lead.tfv6.future_collision import future_collision_label
from scripts.p4.audit_b3b_local_paths import selected_actor_frames
from scripts.p4.eval_b3b_kinematic_oracle import rollout


def load_current_speeds(feature_dir: Path, keys: np.ndarray) -> np.ndarray:
    """Join sensor speeds by exact frame key; never reuse another pass's target."""
    requested = {str(key): index for index, key in enumerate(keys)}
    if len(requested) != len(keys):
        raise ValueError("duplicate B3b frame keys")
    speeds = np.full(len(keys), np.nan, dtype=np.float32)
    for path in sorted(feature_dir.glob("velocity_features_*.npz")):
        with np.load(path, allow_pickle=False) as file:
            for key, speed in zip(file["keys"], file["current_speed"], strict=True):
                index = requested.get(str(key))
                if index is None:
                    continue
                if np.isfinite(speeds[index]):
                    raise ValueError(f"duplicate feature frame key: {key}")
                speeds[index] = float(speed)
        if np.isfinite(speeds).all():
            break
    if not np.isfinite(speeds).all() or (speeds < 0).any():
        raise ValueError(f"missing/invalid speeds: {int((~np.isfinite(speeds)).sum())}")
    return speeds


def eligible_actions(
    *,
    route_valid: np.ndarray,
    direction_ok: np.ndarray,
    corridor_cost: np.ndarray,
    route_length: np.ndarray,
    target_valid: np.ndarray,
    targets: np.ndarray,
    progress: np.ndarray,
    raw_target: float,
    max_corridor_cost: float,
    max_progress_loss_m: float,
) -> np.ndarray:
    """Preregistered path/task and no-speedup screen, without GT safety."""
    raw_progress = float(progress[0])
    path_ok = route_valid & direction_ok & (corridor_cost <= max_corridor_cost)
    action_ok = (
        target_valid
        & (targets <= raw_target + 0.01)
        & (progress >= raw_progress - max_progress_loss_m)
    )
    return (
        path_ok[:, None]
        & action_ok[None, :]
        & (progress[None, :] <= route_length[:, None] + 0.01)
    )


def choose_safe(
    eligible: np.ndarray,
    collision: np.ndarray,
    track_error: np.ndarray,
    progress: np.ndarray,
    winner: int,
    max_track_error_m: float,
    max_path_count: int,
) -> tuple[int, int] | None:
    ok = eligible.copy()
    ok[max_path_count:] = False
    ok &= ~collision & (track_error <= max_track_error_m)
    ok[winner, 0] = False
    if not ok.any():
        return None
    flat = int(np.where(ok, progress[None, :], -np.inf).argmax())
    return tuple(map(int, np.unravel_index(flat, ok.shape)))


def evaluate(
    report_path: Path,
    feature_dir: Path,
    output: Path,
    *,
    max_progress_loss_m: float = 2.0,
    max_track_error_m: float = 1.0,
    max_yaw_rate_deg_s: float = 30.0,
    lookahead_m: float = 2.0,
    corridor_thresholds: tuple[float, ...] = (0.1, 0.3),
) -> dict:
    if max_progress_loss_m < 0 or max_track_error_m <= 0:
        raise ValueError("invalid progress-loss or tracking-error threshold")
    if not corridor_thresholds or any(x < 0 for x in corridor_thresholds):
        raise ValueError("corridor thresholds must be nonnegative")
    report = json.loads(report_path.read_text())
    with np.load(report["frame_records"], allow_pickle=False) as file:
        arrays = {name: file[name] for name in file.files}
    required = {
        "key", "routes", "winner", "raw_target_speed", "route_valid",
        "direction_ok", "corridor_cost", "route_length",
    }
    if not required.issubset(arrays):
        raise ValueError(f"missing B3b fields: {required - arrays.keys()}")
    n, path_count = arrays["routes"].shape[:2]
    original_path_count = int(report["original_path_count"])
    if not 0 < original_path_count <= path_count:
        raise ValueError("invalid original path count")
    speeds = load_current_speeds(feature_dir, arrays["key"])
    actors = selected_actor_frames(
        Path(report["future_cache_dir"]), set(map(str, arrays["key"]))
    )
    action_count = len(ACTION_NAMES)
    targets = np.zeros((n, action_count), dtype=np.float32)
    valid = np.zeros((n, action_count), dtype=bool)
    progress = np.zeros((n, action_count), dtype=np.float32)
    raw_collision = np.zeros(n, dtype=bool)
    collision = np.ones((n, path_count, action_count), dtype=bool)
    track_error = np.full(collision.shape, np.inf, dtype=np.float32)
    evaluated = np.zeros(collision.shape, dtype=bool)
    dynamics = Dynamics()
    config = ExpertConfig()
    safety_margin = float(report["policy"]["safety_margin_m"])

    def test(row: int, arm: int, action: int, interval_speed: np.ndarray) -> None:
        xy, yaw, error = rollout(
            arrays["routes"][row, arm], interval_speed,
            max_yaw_rate_deg_s=max_yaw_rate_deg_s, lookahead_m=lookahead_m,
        )
        collision[row, arm, action] = future_collision_label(
            xy, yaw, actors[str(arrays["key"][row])],
            safety_margin_m=safety_margin, include_class_ids=(1, 2),
        ).collision
        track_error[row, arm, action] = error
        evaluated[row, arm, action] = True

    for row in tqdm(range(n), desc="PID-joint raw", unit="frame"):
        targets[row], valid[row] = scalar_actions(float(arrays["raw_target_speed"][row]))
        win = int(arrays["winner"][row])
        if not 0 <= win < path_count:
            raise ValueError(f"invalid winner at row {row}")
        raw = rollout_scalar_target(
            float(speeds[row]), float(targets[row, 0]),
            dynamics=dynamics, expert_config=config,
        )
        progress[row, 0] = raw.progress_m
        intervals = np.diff(np.r_[0.0, raw.xy_distance_m]) / float(FUTURE_TIMES_S[0])
        test(row, win, 0, intervals)
        raw_collision[row] = collision[row, win, 0]

    for row in tqdm(np.flatnonzero(raw_collision), desc="PID-joint counterfactuals", unit="frame"):
        win = int(arrays["winner"][row])
        intervals_by_action = []
        for action in range(action_count):
            if valid[row, action]:
                outcome = rollout_scalar_target(
                    float(speeds[row]), float(targets[row, action]),
                    dynamics=dynamics, expert_config=config,
                )
                progress[row, action] = outcome.progress_m
                intervals_by_action.append(
                    np.diff(np.r_[0.0, outcome.xy_distance_m]) / float(FUTURE_TIMES_S[0])
                )
            else:
                intervals_by_action.append(None)
        # Compute the union of all preregistered corridor screens once.
        eligible = eligible_actions(
            route_valid=arrays["route_valid"][row],
            direction_ok=arrays["direction_ok"][row],
            corridor_cost=arrays["corridor_cost"][row],
            route_length=arrays["route_length"][row],
            target_valid=valid[row], targets=targets[row], progress=progress[row],
            raw_target=float(targets[row, 0]),
            max_corridor_cost=max(corridor_thresholds),
            max_progress_loss_m=max_progress_loss_m,
        )
        for arm, action in np.argwhere(eligible):
            if arm == win and action == 0:
                continue
            test(row, int(arm), int(action), intervals_by_action[action])

    results = {}
    selected = {}
    raw_track_error = track_error[np.arange(n), arrays["winner"], 0]
    raw_track_qualified = raw_track_error <= max_track_error_m
    for threshold in corridor_thresholds:
        choices = {name: np.full((n, 2), -1, dtype=np.int16)
                   for name in ("fixed_path", "original_paths", "expanded_paths")}
        for row in np.flatnonzero(raw_collision):
            win = int(arrays["winner"][row])
            eligible = eligible_actions(
                route_valid=arrays["route_valid"][row],
                direction_ok=arrays["direction_ok"][row],
                corridor_cost=arrays["corridor_cost"][row],
                route_length=arrays["route_length"][row],
                target_valid=valid[row], targets=targets[row], progress=progress[row],
                raw_target=float(targets[row, 0]),
                max_corridor_cost=threshold,
                max_progress_loss_m=max_progress_loss_m,
            ) & evaluated[row]
            fixed = eligible.copy()
            fixed[np.arange(path_count) != win] = False
            for name, mask, limit in (
                ("fixed_path", fixed, path_count),
                ("original_paths", eligible, original_path_count),
                ("expanded_paths", eligible, path_count),
            ):
                choice = choose_safe(
                    mask, collision[row], track_error[row], progress[row],
                    win, max_track_error_m, limit,
                )
                if choice is not None:
                    choices[name][row] = choice
        metrics = {}
        for name, choice in choices.items():
            rescued = choice[:, 0] >= 0
            picked_action = choice[rescued, 1]
            metrics[name] = {
                "rescued_frames": int(rescued.sum()),
                "rescue_given_raw_collision": float(rescued.sum() / max(raw_collision.sum(), 1)),
                "distinct_routes": len({str(key).rsplit("__", 1)[0]
                                        for key in arrays["key"][rescued]}),
                "path_switches": int((choice[rescued, 0] != arrays["winner"][rescued]).sum()),
                "moving_rescues": int((targets[rescued, picked_action] > 0.01).sum()),
                "action_counts": {action: int((picked_action == idx).sum())
                                  for idx, action in enumerate(ACTION_NAMES)},
            }
        fixed = choices["fixed_path"][:, 0] >= 0
        original = choices["original_paths"][:, 0] >= 0
        expanded = choices["expanded_paths"][:, 0] >= 0
        metrics["joint_beyond_fixed_count"] = int((original & ~fixed).sum())
        metrics["local_path_incremental_count"] = int((expanded & ~original).sum())
        local_incremental = expanded & ~original
        local_actions = choices["expanded_paths"][local_incremental, 1]
        metrics["local_path_incremental_baseline_speed_count"] = int(
            (local_actions == 0).sum()
        )
        metrics["local_path_incremental_routes"] = len({
            str(key).rsplit("__", 1)[0]
            for key in arrays["key"][local_incremental]
        })
        metrics["raw_unsafe_within_track_error_count"] = int(
            (raw_collision & raw_track_qualified).sum()
        )
        metrics["expanded_rescue_with_raw_track_qualified_count"] = int(
            (expanded & raw_track_qualified).sum()
        )
        # Gate is for candidate coverage only, not permission to deploy a model.
        metrics["coverage_gate"] = (
            "pass" if raw_collision.sum() >= 50
            and metrics["expanded_paths"]["rescue_given_raw_collision"] >= 0.20
            and metrics["local_path_incremental_count"] >= 10
            else "fail"
        )
        results[f"corridor_{threshold:g}"] = metrics
        for name, choice in choices.items():
            selected[f"corridor_{threshold:g}_{name}_choice"] = choice

    output.parent.mkdir(parents=True, exist_ok=True)
    frame_path = output.with_suffix(".frames.npz")
    np.savez_compressed(
        frame_path, key=arrays["key"], current_speed=speeds,
        raw_collision=raw_collision, targets=targets, valid=valid, progress=progress,
        candidate_collision=collision, evaluated=evaluated, track_error=track_error,
        **selected,
    )
    result = {
        "source": str(report_path.resolve()),
        "features": str(feature_dir.resolve()),
        "frames": n,
        "raw_collision_count": int(raw_collision.sum()),
        "raw_collision_rate": float(raw_collision.mean()),
        "raw_within_track_error_count": int(raw_track_qualified.sum()),
        "original_B3b_raw_collision_count": int(arrays["raw_collision"].sum()),
        "raw_overlap": int((raw_collision & arrays["raw_collision"]).sum()),
        "policy": {
            "horizon_s": float(FUTURE_TIMES_S[-1]),
            "max_progress_loss_m": max_progress_loss_m,
            "max_track_error_m": max_track_error_m,
            "max_yaw_rate_deg_s": max_yaw_rate_deg_s,
            "lookahead_m": lookahead_m,
            "corridor_thresholds": corridor_thresholds,
            "no_speedups": True,
            "safety_margin_m": safety_margin,
        },
        "caveat": "GT future actors and expert direction; simplified lateral follower and approximate vehicle dynamics, not CARLA or receding-horizon LEAD",
        "results": results,
        "frame_records": str(frame_path.resolve()),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=root / "outputs/local_training/p5_stepB3b_joint_oracle/joint_nested_k16_k32_local_8gpu.json")
    parser.add_argument("--features", type=Path, default=root / "outputs/local_training/p5_stepB3a_v2_scene_scorer/features/heldout")
    parser.add_argument("--out", type=Path, default=root / "outputs/local_training/p5_stepB3b_joint_oracle/pid_joint_coverage_5000.json")
    parser.add_argument("--max-progress-loss-m", type=float, default=2.0)
    parser.add_argument("--max-track-error-m", type=float, default=1.0)
    args = parser.parse_args()
    result = evaluate(
        args.report, args.features, args.out,
        max_progress_loss_m=args.max_progress_loss_m,
        max_track_error_m=args.max_track_error_m,
    )
    print(f"raw collisions: {result['raw_collision_count']}/{result['frames']}")
    for threshold, metrics in result["results"].items():
        print(f"{threshold}: fixed={metrics['fixed_path']['rescued_frames']} "
              f"original={metrics['original_paths']['rescued_frames']} "
              f"expanded={metrics['expanded_paths']['rescued_frames']} "
              f"local+={metrics['local_path_incremental_count']} "
              f"gate={metrics['coverage_gate']}")
    print(f"wrote {args.out} and {result['frame_records']}")


if __name__ == "__main__":
    main()
