"""Recheck B3b Path x Velocity candidates with a yaw-continuous rollout.

GT future actors still make this an oracle, not a deployable closed-loop policy.
The rollout is a simple bounded-yaw unicycle path follower, not LEAD's PID
controller or a full bicycle model. Ego positions and yaws are integrated
together before SAT collision testing.
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
from lead.tfv6.future_collision import future_collision_label
from scripts.p4.audit_b3b_local_paths import selected_actor_frames
from scripts.p4.eval_b3b_joint_oracle import candidate_eligibility, choose_oracle, summarize


def prepared_route(route: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(route, dtype=np.float32).reshape(-1, 2)
    if np.linalg.norm(points[0]) > 1e-4:
        points = np.concatenate((np.zeros((1, 2), dtype=np.float32), points))
    keep = np.concatenate((
        [True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-5
    ))
    points = points[keep]
    if len(points) == 1:
        points = np.concatenate((points, points + [1e-3, 0.0]))
    arc = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    ))
    return points, arc


def route_point(prepared: tuple[np.ndarray, np.ndarray], distance_m: float) -> np.ndarray:
    points, arc = prepared
    distance = max(float(distance_m), 0.0)
    if distance > arc[-1]:
        tangent = (points[-1] - points[-2]) / max(float(arc[-1] - arc[-2]), 1e-6)
        return (points[-1] + (distance - arc[-1]) * tangent).astype(np.float32)
    return np.array([
        np.interp(distance, arc, points[:, 0]),
        np.interp(distance, arc, points[:, 1]),
    ], dtype=np.float32)


def rollout(
    route: np.ndarray,
    interval_speeds: np.ndarray,
    *,
    max_yaw_rate_deg_s: float = 30.0,
    lookahead_m: float = 2.0,
    interval_s: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Track a geometric path from ego pose (0,0,0) with consistent pose integration."""

    if max_yaw_rate_deg_s <= 0 or lookahead_m <= 0 or interval_s <= 0:
        raise ValueError("yaw rate, lookahead and interval must be positive")
    speeds = np.asarray(interval_speeds, dtype=np.float32).reshape(-1)
    if not np.isfinite(speeds).all() or np.any(speeds < 0):
        raise ValueError("interval speeds must be finite and nonnegative")
    path = prepared_route(route)
    positions = np.zeros((len(speeds), 2), dtype=np.float32)
    yaws = np.zeros(len(speeds), dtype=np.float32)
    xy = np.zeros(2, dtype=np.float64)
    yaw = 0.0
    desired_distance = 0.0
    max_error = 0.0
    max_increment = np.deg2rad(max_yaw_rate_deg_s) * interval_s
    for step, speed in enumerate(speeds):
        desired_distance += float(speed) * interval_s
        target = route_point(path, desired_distance + lookahead_m)
        desired_yaw = np.arctan2(float(target[1] - xy[1]), float(target[0] - xy[0]))
        yaw_delta = np.arctan2(np.sin(desired_yaw - yaw), np.cos(desired_yaw - yaw))
        if speed <= 1e-3:
            yaw_delta = 0.0  # a stopped car cannot rotate in place
        yaw_delta = float(np.clip(yaw_delta, -max_increment, max_increment))
        midpoint_yaw = yaw + 0.5 * yaw_delta
        xy += float(speed) * interval_s * np.array([
            np.cos(midpoint_yaw), np.sin(midpoint_yaw)
        ])
        yaw += yaw_delta
        positions[step] = xy
        yaws[step] = yaw
        error = np.linalg.norm(xy - route_point(path, desired_distance))
        max_error = max(max_error, float(error))
    return positions, yaws, max_error


def evaluate(
    report_path: Path,
    output: Path,
    *,
    max_yaw_rate_deg_s: float = 30.0,
    lookahead_m: float = 2.0,
    track_error_thresholds: tuple[float, ...] = (1.0, 2.0),
) -> dict:
    report = json.loads(report_path.read_text())
    with np.load(report["frame_records"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = {"routes", "candidate_velocity", "candidate_collision"}
    if not required.issubset(arrays):
        raise ValueError("input must be the nested B3b report with saved route geometry")
    if any(value <= 0 for value in track_error_thresholds):
        raise ValueError("tracking-error thresholds must be positive")
    n, k_count, m_count = arrays["candidate_collision"].shape
    if arrays["candidate_velocity"].shape != (n, m_count, len(FUTURE_TIMES_S)):
        raise ValueError("candidate velocity and collision lattice do not match")
    actors = selected_actor_frames(
        Path(report["future_cache_dir"]), set(map(str, arrays["key"]))
    )
    collision = np.ones((n, k_count, m_count), dtype=bool)
    evaluated = np.zeros_like(collision)
    track_error = np.full((n, k_count, m_count), np.inf, dtype=np.float32)
    old_raw = arrays["raw_collision"]
    raw_collision = np.zeros(n, dtype=bool)
    safety_margin = report["policy"]["safety_margin_m"]

    def test_candidate(row: int, arm: int, mode: int) -> None:
        positions, yaws, error = rollout(
            arrays["routes"][row, arm], arrays["candidate_velocity"][row, mode],
            max_yaw_rate_deg_s=max_yaw_rate_deg_s, lookahead_m=lookahead_m,
        )
        collision[row, arm, mode] = future_collision_label(
            positions, yaws, actors[str(arrays["key"][row])],
            safety_margin_m=safety_margin, include_class_ids=(1, 2),
        ).collision
        track_error[row, arm, mode] = error
        evaluated[row, arm, mode] = True

    for row in tqdm(range(n), desc="kinematic raw collision"):
        win = int(arrays["winner"][row])
        test_candidate(row, win, 0)
        raw_collision[row] = collision[row, win, 0]
    for row in tqdm(np.flatnonzero(raw_collision), desc="kinematic counterfactuals"):
        win = int(arrays["winner"][row])
        eligible = candidate_eligibility(
            arrays["route_valid"][row], arrays["direction_ok"][row],
            arrays["corridor_cost"][row], arrays["route_length"][row],
            arrays["candidate_valid"][row], arrays["candidate_progress"][row],
            arrays["candidate_pid_target"][row],
            raw_progress=float(arrays["raw_progress"][row]),
            raw_target_speed=float(arrays["raw_target_speed"][row]),
            max_corridor_cost=report["policy"]["max_off_corridor_cost"],
            min_progress_ratio=report["policy"]["min_progress_ratio"],
            conservative=False,
        )
        for arm, mode in np.argwhere(eligible):
            if arm == win and mode == 0:
                continue
            test_candidate(int(row), int(arm), int(mode))

    scopes = {
        "all": np.ones(n, dtype=bool),
        "multi": arrays["multi"],
        "common_old_and_kinematic_unsafe": old_raw & raw_collision,
    }
    results = {}
    selected = {}
    for policy_name, conservative in (("reachable", False), ("conservative", True)):
        results[policy_name] = {}
        for error_name, max_error in (
            ("unrestricted_tracking", float("inf")),
            *((f"max_track_error_{value:g}m", value) for value in track_error_thresholds),
        ):
            fixed_rescue = np.zeros(n, dtype=bool)
            joint_rescue = np.zeros(n, dtype=bool)
            expanded_rescue = np.zeros(n, dtype=bool)
            joint_arm = arrays["winner"].copy()
            joint_mode = np.zeros(n, dtype=np.int16)
            expanded_arm = arrays["winner"].copy()
            expanded_mode = np.zeros(n, dtype=np.int16)
            for row in np.flatnonzero(raw_collision):
                win = int(arrays["winner"][row])
                eligible = candidate_eligibility(
                    arrays["route_valid"][row], arrays["direction_ok"][row],
                    arrays["corridor_cost"][row], arrays["route_length"][row],
                    arrays["candidate_valid"][row], arrays["candidate_progress"][row],
                    arrays["candidate_pid_target"][row],
                    raw_progress=float(arrays["raw_progress"][row]),
                    raw_target_speed=float(arrays["raw_target_speed"][row]),
                    max_corridor_cost=report["policy"]["max_off_corridor_cost"],
                    min_progress_ratio=report["policy"]["min_progress_ratio"],
                    conservative=conservative,
                )
                eligible &= evaluated[row] & (track_error[row] <= max_error)
                progress = np.broadcast_to(arrays["candidate_progress"][row], eligible.shape)
                fixed = choose_oracle(
                    True, collision[row], eligible, progress, win, joint=False
                )
                original_eligible = eligible.copy()
                original_eligible[report["original_path_count"]:] = False
                joint = choose_oracle(
                    True, collision[row], original_eligible, progress, win, joint=True
                )
                expanded = choose_oracle(
                    True, collision[row], eligible, progress, win, joint=True
                )
                fixed_rescue[row] = fixed != (win, 0)
                joint_rescue[row] = joint != (win, 0)
                expanded_rescue[row] = expanded != (win, 0)
                joint_arm[row], joint_mode[row] = joint
                expanded_arm[row], expanded_mode[row] = expanded
            results[policy_name][error_name] = {}
            for scope_name, scope in scopes.items():
                scope_results = {}
                for candidate_name, rescue, arm, mode in (
                    ("original_paths", joint_rescue, joint_arm, joint_mode),
                    ("expanded_paths", expanded_rescue, expanded_arm, expanded_mode),
                ):
                    progress = np.where(
                        rescue,
                        arrays["candidate_progress"][np.arange(n), mode],
                        arrays["raw_progress"],
                    )
                    metrics = summarize(
                        raw_collision=raw_collision, fixed_rescue=fixed_rescue,
                        joint_rescue=rescue, joint_arm=arm,
                        winner_arm=arrays["winner"],
                        raw_progress=arrays["raw_progress"],
                        joint_progress=progress, scope=scope,
                    )
                    metrics["moving_rescue_at_least_1m_count"] = int((
                        raw_collision & scope & rescue & (progress >= 1.0)
                    ).sum())
                    scope_results[candidate_name] = metrics
                scope_results["incremental_local_rescue_count"] = int((
                    raw_collision & scope & expanded_rescue & ~joint_rescue
                ).sum())
                scope_results["incremental_local_moving_at_least_1m_count"] = int((
                    raw_collision & scope & expanded_rescue & ~joint_rescue
                    & (arrays["candidate_progress"][np.arange(n), expanded_mode] >= 1.0)
                ).sum())
                results[policy_name][error_name][scope_name] = scope_results
            if policy_name == "conservative" and error_name == "max_track_error_1m":
                selected = {
                    "fixed_rescue": fixed_rescue,
                    "original_path_rescue": joint_rescue,
                    "expanded_path_rescue": expanded_rescue,
                    "expanded_arm": expanded_arm,
                    "expanded_mode": expanded_mode,
                }
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_path = output.with_suffix(".frames.npz")
    np.savez_compressed(
        frame_path, key=arrays["key"], old_raw_collision=old_raw,
        kinematic_raw_collision=raw_collision, candidate_collision=collision,
        collision_evaluated=evaluated, max_tracking_error_m=track_error,
        **selected,
    )
    output_report = {
        "source": str(report_path.resolve()),
        "n_frames": n,
        "candidate_shape": [k_count, m_count],
        "max_yaw_rate_deg_s": max_yaw_rate_deg_s,
        "lookahead_m": lookahead_m,
        "rollout": "bounded-yaw unicycle, midpoint pose integration, current ego yaw=0",
        "caveat": "GT future actors; simplified path follower, not LEAD PID or CARLA closed loop",
        "raw_collision": {
            "original_count": int(old_raw.sum()),
            "kinematic_count": int(raw_collision.sum()),
            "common_count": int((old_raw & raw_collision).sum()),
            "original_only_count": int((old_raw & ~raw_collision).sum()),
            "kinematic_only_count": int((raw_collision & ~old_raw).sum()),
            "raw_max_track_error_median": float(np.median(
                track_error[np.arange(n), arrays["winner"], 0]
            )),
        },
        "results": results,
        "frame_records": str(frame_path.resolve()),
    }
    output.write_text(json.dumps(output_report, indent=2))
    return output_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-yaw-rate-deg-s", type=float, default=30.0)
    parser.add_argument("--lookahead-m", type=float, default=2.0)
    parser.add_argument("--track-error-thresholds", default="1.0,2.0")
    args = parser.parse_args()
    thresholds = tuple(float(value) for value in args.track_error_thresholds.split(","))
    result = evaluate(
        args.report, args.out,
        max_yaw_rate_deg_s=args.max_yaw_rate_deg_s,
        lookahead_m=args.lookahead_m,
        track_error_thresholds=thresholds,
    )
    print(f"raw collision old/rollout: {result['raw_collision']['original_count']}/"
          f"{result['raw_collision']['kinematic_count']}")
    for policy in ("reachable", "conservative"):
        for track, scopes in result["results"][policy].items():
            all_frames = scopes["all"]
            print(
                f"{policy} {track}: original={all_frames['original_paths']['joint_rescue_count']} "
                f"expanded={all_frames['expanded_paths']['joint_rescue_count']} "
                f"local+={all_frames['incremental_local_rescue_count']}"
            )
    print(f"wrote {args.out} and {result['frame_records']}")


if __name__ == "__main__":
    main()
