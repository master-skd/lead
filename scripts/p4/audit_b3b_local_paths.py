"""Screen GT-oracle local-Path rescues with geometric and intent-corridor proxies.

This cannot establish closed-loop trackability or true road occupancy. It flags
obviously sharp or off-predicted-corridor paths for closer inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.p4.eval_b3b_joint_oracle import candidate_eligibility, choose_oracle
from lead.data_loader.future_actor_cache import unpack_future_actor_frame
from lead.tfv6.future_collision import future_collision_label, interpolate_route_by_distance


def yaw_rate_limited(yaws: np.ndarray, max_rate_deg_s: float, interval_s: float = 0.25) -> np.ndarray:
    """Start at the current ego heading (zero) and cap each yaw increment."""

    limit = np.deg2rad(max_rate_deg_s) * interval_s
    result = np.empty_like(yaws, dtype=np.float32)
    previous = 0.0
    for step, target in enumerate(yaws):
        delta = np.arctan2(np.sin(float(target) - previous), np.cos(float(target) - previous))
        previous += float(np.clip(delta, -limit, limit))
        result[step] = previous
    return result


def selected_actor_frames(cache_dir: Path, keys: set[str]) -> dict:
    if not keys:
        return {}
    found = {}
    for path in sorted(cache_dir.glob("future_actors_*.npz")):
        with np.load(path, allow_pickle=False) as shard:
            for index, key in enumerate(shard["keys"]):
                key = str(key)
                if key in keys:
                    found[key] = unpack_future_actor_frame(shard, index)
        if len(found) == len(keys):
            break
    missing = keys - found.keys()
    if missing:
        raise KeyError(f"missing {len(missing)} selected future-actor frames")
    return found


def add_base_vocabulary_selection(arrays: dict[str, np.ndarray], report: dict) -> int | None:
    """Re-select the preserved first source vocab without another model pass."""

    sidecar = Path(report["vocabulary"]).with_suffix(".json")
    if not sidecar.exists():
        return None
    metadata = json.loads(sidecar.read_text())
    if not metadata.get("prefix_preserved") or not metadata.get("sources"):
        return None
    base_count = int(metadata["sources"][0]["shape"][0])
    if base_count <= 0 or base_count >= arrays["candidate_valid"].shape[1] - 1:
        raise ValueError("invalid nested-vocabulary base count")
    n = len(arrays["key"])
    for policy, conservative in (("reachable", False), ("conservative", True)):
        joint_rescue = np.zeros(n, dtype=bool)
        expanded_rescue = np.zeros(n, dtype=bool)
        expanded_arm = arrays["winner"].copy()
        expanded_mode = np.zeros(n, dtype=np.int16)
        for row in np.flatnonzero(arrays["raw_collision"]):
            winner = int(arrays["winner"][row])
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
            eligible[:, base_count + 1:] = False
            progress = np.broadcast_to(
                arrays["candidate_progress"][row], eligible.shape
            )
            original = eligible.copy()
            original[report["original_path_count"]:] = False
            joint = choose_oracle(
                True, arrays["candidate_collision"][row], original,
                progress, winner, joint=True,
            )
            expanded = choose_oracle(
                True, arrays["candidate_collision"][row], eligible,
                progress, winner, joint=True,
            )
            joint_rescue[row] = joint != (winner, 0)
            expanded_rescue[row] = expanded != (winner, 0)
            expanded_arm[row], expanded_mode[row] = expanded
        prefix = f"{policy}_base_k{base_count}"
        arrays[f"{prefix}_joint_rescue"] = joint_rescue
        arrays[f"{prefix}_expanded_rescue"] = expanded_rescue
        arrays[f"{prefix}_expanded_arm"] = expanded_arm
        arrays[f"{prefix}_expanded_mode"] = expanded_mode
    return base_count


def max_prefix_curvature(route: np.ndarray, distance_m: float) -> float:
    """Maximum three-point curvature along the first ``distance_m`` of a polyline."""

    points = np.asarray(route, dtype=np.float64).reshape(-1, 2)
    if np.linalg.norm(points[0]) > 1e-4:
        points = np.concatenate((np.zeros((1, 2)), points))
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.concatenate(([True], lengths > 1e-4))]
    if len(points) < 3 or distance_m <= 0:
        return 0.0
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    a = points[1:-1] - points[:-2]
    b = points[2:] - points[1:-1]
    c = points[2:] - points[:-2]
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1)
    curvature = 2.0 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / np.maximum(denom, 1e-8)
    visited = arc[1:-1] <= float(distance_m) + 1e-6
    return float(curvature[visited].max()) if visited.any() else 0.0


def prefix_corridor_max(route: np.ndarray, cost: np.ndarray, distance_m: float) -> float:
    points = np.asarray(route, dtype=np.float64).reshape(-1, 2)
    arc = np.cumsum(np.linalg.norm(np.diff(np.concatenate((np.zeros((1, 2)), points)), axis=0), axis=1))
    keep = arc <= distance_m + 1e-6
    keep[0] = True
    return float(np.asarray(cost)[keep].max())


def quantiles(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"median": None, "p90": None, "max": None}
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def audit(report_path: Path, output: Path) -> dict:
    report = json.loads(report_path.read_text())
    with np.load(report["frame_records"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = {"routes", "corridor_points", "candidate_velocity"}
    if not required.issubset(arrays):
        raise ValueError("report predates geometric-record extraction; rerun the oracle")
    base_count = add_base_vocabulary_selection(arrays, report)
    n = len(arrays["key"])
    output_report = {
        "source": str(report_path.resolve()),
        "n_frames": n,
        "interpretation": (
            "Curvature and predicted-intent-corridor proxies only; not a proof "
            "of closed-loop controllability or true drivable-area containment"
        ),
        "policies": {},
    }
    policies = ["reachable", "conservative"]
    if base_count is not None:
        policies += [f"reachable_base_k{base_count}", f"conservative_base_k{base_count}"]
    selected_rows = set()
    for policy in policies:
        incremental = (
            arrays["raw_collision"] & arrays[f"{policy}_expanded_rescue"]
            & ~arrays[f"{policy}_joint_rescue"]
        )
        selected_rows.update(np.flatnonzero(incremental).tolist())
    actors = selected_actor_frames(
        Path(report["future_cache_dir"]),
        {str(arrays["key"][row]) for row in selected_rows},
    )
    for policy in policies:
        if f"{policy}_expanded_rescue" not in arrays:
            continue
        incremental = (
            arrays["raw_collision"] & arrays[f"{policy}_expanded_rescue"]
            & ~arrays[f"{policy}_joint_rescue"]
        )
        ids = np.flatnonzero(incremental)
        modes = arrays[f"{policy}_expanded_mode"][ids]
        progress = arrays["candidate_progress"][ids, modes]
        moving = progress >= 1.0
        moving_ids = ids[moving]
        chosen_curvature = []
        raw_curvature = []
        chosen_corridor = []
        raw_corridor = []
        lateral_accel_upper_proxy = []
        initial_heading_delta_deg = []
        yaw_limited_30_collision = []
        yaw_limited_60_collision = []
        original_yaw_collision = []
        for row in ids:
            arm = int(arrays[f"{policy}_expanded_arm"][row])
            mode = int(arrays[f"{policy}_expanded_mode"][row])
            route = arrays["routes"][row, arm]
            distances = arrays["candidate_velocity"][row, mode].cumsum() * 0.25
            positions, yaws = interpolate_route_by_distance(route, distances, extrapolate=True)
            actor_frame = actors[str(arrays["key"][row])]
            collision_kwargs = dict(
                safety_margin_m=report["policy"]["safety_margin_m"],
                include_class_ids=(1, 2),
            )
            original_yaw_collision.append(future_collision_label(
                positions, yaws, actor_frame, **collision_kwargs
            ).collision)
            yaw_limited_30_collision.append(future_collision_label(
                positions, yaw_rate_limited(yaws, 30), actor_frame, **collision_kwargs
            ).collision)
            yaw_limited_60_collision.append(future_collision_label(
                positions, yaw_rate_limited(yaws, 60), actor_frame, **collision_kwargs
            ).collision)
        for row in moving_ids:
            arm = int(arrays[f"{policy}_expanded_arm"][row])
            mode = int(arrays[f"{policy}_expanded_mode"][row])
            distance = float(arrays["candidate_progress"][row, mode])
            winner = int(arrays["winner"][row])
            chosen_curvature.append(max_prefix_curvature(arrays["routes"][row, arm], distance))
            raw_curvature.append(max_prefix_curvature(arrays["routes"][row, winner], distance))
            chosen_corridor.append(prefix_corridor_max(
                arrays["routes"][row, arm], arrays["corridor_points"][row, arm], distance
            ))
            raw_corridor.append(prefix_corridor_max(
                arrays["routes"][row, winner], arrays["corridor_points"][row, winner], distance
            ))
            max_speed = float(arrays["candidate_velocity"][row, mode].max())
            lateral_accel_upper_proxy.append(max_speed**2 * chosen_curvature[-1])
            route = arrays["routes"][row, arm]
            raw_route = arrays["routes"][row, winner]
            chosen_heading = np.arctan2(route[0, 1], route[0, 0])
            raw_heading = np.arctan2(raw_route[0, 1], raw_route[0, 0])
            heading_delta = np.arctan2(
                np.sin(chosen_heading - raw_heading),
                np.cos(chosen_heading - raw_heading),
            )
            initial_heading_delta_deg.append(abs(float(np.rad2deg(heading_delta))))
        chosen_curvature = np.asarray(chosen_curvature)
        raw_curvature = np.asarray(raw_curvature)
        chosen_corridor = np.asarray(chosen_corridor)
        raw_corridor = np.asarray(raw_corridor)
        lateral_accel_upper_proxy = np.asarray(lateral_accel_upper_proxy)
        initial_heading_delta_deg = np.asarray(initial_heading_delta_deg)
        output_report["policies"][policy] = {
            "incremental_local_rescue_frames": int(len(ids)),
            "moving_at_least_1m": int(moving.sum()),
            "near_stationary_under_1m": int((~moving).sum()),
            "moving_raw_speed_mode_count": int((arrays[f"{policy}_expanded_mode"][moving_ids] == 0).sum()),
            "selected_curvature_1_per_m": quantiles(chosen_curvature),
            "raw_path_curvature_1_per_m": quantiles(raw_curvature),
            "selected_curvature_above_raw_count": int((chosen_curvature > raw_curvature + 1e-4).sum()),
            "selected_curvature_above_0p2_count": int((chosen_curvature > 0.2).sum()),
            "selected_curvature_above_0p35_count": int((chosen_curvature > 0.35).sum()),
            "speed_squared_times_max_curvature_upper_proxy_mps2": quantiles(lateral_accel_upper_proxy),
            "upper_proxy_above_3mps2_count": int((lateral_accel_upper_proxy > 3.0).sum()),
            "initial_heading_delta_from_raw_deg": quantiles(initial_heading_delta_deg),
            "original_yaw_recomputed_collision_count": int(sum(original_yaw_collision)),
            "yaw_rate_limited_30deg_s_collision_count": int(sum(yaw_limited_30_collision)),
            "yaw_rate_limited_60deg_s_collision_count": int(sum(yaw_limited_60_collision)),
            "moving_yaw_rate_limited_30deg_s_collision_count": int(np.sum(
                np.asarray(yaw_limited_30_collision, dtype=bool)[moving]
            )),
            "near_stationary_yaw_rate_limited_30deg_s_collision_count": int(np.sum(
                np.asarray(yaw_limited_30_collision, dtype=bool)[~moving]
            )),
            "selected_predicted_corridor_point_max": quantiles(chosen_corridor),
            "raw_predicted_corridor_point_max": quantiles(raw_corridor),
            "selected_corridor_point_above_0p1_count": int((chosen_corridor > 0.1).sum()),
            "selected_corridor_point_above_0p25_count": int((chosen_corridor > 0.25).sum()),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_report, indent=2))
    return output_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.report, args.out)
    for policy, metrics in result["policies"].items():
        print(
            f"{policy}: incremental={metrics['incremental_local_rescue_frames']}, "
            f"moving={metrics['moving_at_least_1m']}, "
            f"curvature>0.2={metrics['selected_curvature_above_0p2_count']}, "
            f"corridor-point>0.1={metrics['selected_corridor_point_above_0p1_count']}, "
            f"yaw30-coll={metrics['yaw_rate_limited_30deg_s_collision_count']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
