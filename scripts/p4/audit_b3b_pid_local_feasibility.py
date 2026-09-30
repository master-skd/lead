"""Audit local-Path oracle rescues against predicted corridor and dynamics proxies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lead.data_loader.future_actor_cache import FUTURE_TIMES_S
from lead.expert.config_expert import ExpertConfig
from lead.tfv6.control_aware_velocity_oracle import Dynamics, rollout_scalar_target
from scripts.p4.audit_b3b_local_paths import (
    max_prefix_curvature, prefix_corridor_max, quantiles,
)
from scripts.p4.eval_b3b_kinematic_oracle import rollout


def hdmap_path(root: Path, key: str) -> Path:
    try:
        scenario, route, frame = key.rsplit("__", 2)
    except ValueError as error:
        raise ValueError(f"unrecognized B3b key: {key}") from error
    return root / scenario / route / "hdmap" / f"{frame}.png"


def road_fraction(
    hdmap: np.ndarray, positions: np.ndarray, yaws: np.ndarray,
    *, half_width_m: float = 0.8, pixels_per_meter: float = 2.0,
) -> float:
    """Sample the GT static HD-map under a narrow swept-ego footprint.

    Raw ChaffeurNet classes 1/3/4 mean road/solid/broken lane marking.
    This tests static map containment, not traffic rules or dynamic occupancy.
    """
    if hdmap.ndim != 2 or hdmap.shape[0] != hdmap.shape[1]:
        raise ValueError("expected a square, single-channel raw HD-map")
    poses = np.concatenate((np.zeros((1, 2)), np.asarray(positions)), axis=0)
    angles = np.r_[0.0, np.asarray(yaws)]
    center = hdmap.shape[0] / 2
    samples = []
    for step in range(1, len(poses)):
        distance = np.linalg.norm(poses[step] - poses[step - 1])
        subdivisions = max(1, int(np.ceil(distance / 0.25)))
        for fraction in np.linspace(1 / subdivisions, 1, subdivisions):
            xy = poses[step - 1] * (1 - fraction) + poses[step] * fraction
            yaw = angles[step - 1] * (1 - fraction) + angles[step] * fraction
            normal = np.array([-np.sin(yaw), np.cos(yaw)])
            for lateral in (-half_width_m, 0.0, half_width_m):
                point = xy + lateral * normal
                col = int(np.rint(center + point[0] * pixels_per_meter))
                row = int(np.rint(center + point[1] * pixels_per_meter))
                is_road = (
                    0 <= row < hdmap.shape[0] and 0 <= col < hdmap.shape[1]
                    and int(hdmap[row, col]) in (1, 3, 4)
                )
                samples.append(is_road)
    return float(np.mean(samples)) if samples else 0.0


def audit(joint_report: Path, pid_report: Path, output: Path, hdmap_root: Path) -> dict:
    joint = json.loads(joint_report.read_text())
    pid = json.loads(pid_report.read_text())
    with np.load(joint["frame_records"], allow_pickle=False) as archive:
        base = {name: archive[name] for name in archive.files}
    with np.load(pid["frame_records"], allow_pickle=False) as archive:
        actions = {name: archive[name] for name in archive.files}
    if not np.array_equal(base["key"], actions["key"]):
        raise ValueError("joint and PID audit frame keys differ")
    results = {}
    for threshold in pid["policy"]["corridor_thresholds"]:
        prefix = f"corridor_{threshold:g}"
        expanded = actions[f"{prefix}_expanded_paths_choice"]
        original = actions[f"{prefix}_original_paths_choice"]
        incremental = (expanded[:, 0] >= 0) & (original[:, 0] < 0)
        rows = np.flatnonzero(incremental)
        measurements = {name: [] for name in (
            "curvature_1_per_m", "raw_curvature_1_per_m", "corridor_point_max",
            "raw_corridor_point_max", "lateral_accel_proxy_mps2",
            "track_error_m", "raw_track_error_m", "progress_m",
            "gt_road_fraction", "raw_gt_road_fraction",
            "follower_lateral_accel_mps2", "raw_follower_lateral_accel_mps2",
        )}
        details = []
        for row in rows:
            arm, action = map(int, expanded[row])
            winner = int(base["winner"][row])
            progress = float(actions["progress"][row, action])
            route = base["routes"][row, arm]
            raw_route = base["routes"][row, winner]
            curvature = max_prefix_curvature(route, progress)
            raw_curvature = max_prefix_curvature(raw_route, progress)
            point_cost = prefix_corridor_max(
                route, base["corridor_points"][row, arm], progress
            )
            raw_point_cost = prefix_corridor_max(
                raw_route, base["corridor_points"][row, winner], progress
            )
            speed = float(actions["targets"][row, action])
            lateral_accel = speed * speed * curvature
            error = float(actions["track_error"][row, arm, action])
            raw_error = float(actions["track_error"][row, winner, 0])
            map_path = hdmap_path(hdmap_root, str(base["key"][row]))
            hdmap = cv2.imread(str(map_path), cv2.IMREAD_UNCHANGED)
            if hdmap is None:
                raise FileNotFoundError(f"missing GT HD-map {map_path}")
            def sampled_road(candidate_arm: int, candidate_action: int) -> tuple[float, float]:
                outcome = rollout_scalar_target(
                    float(actions["current_speed"][row]),
                    float(actions["targets"][row, candidate_action]),
                    dynamics=Dynamics(), expert_config=ExpertConfig(),
                )
                intervals = np.diff(np.r_[0.0, outcome.xy_distance_m]) / float(FUTURE_TIMES_S[0])
                positions, yaws, _ = rollout(
                    base["routes"][row, candidate_arm], intervals,
                    max_yaw_rate_deg_s=pid["policy"]["max_yaw_rate_deg_s"],
                    lookahead_m=pid["policy"]["lookahead_m"],
                )
                yaw_rate = np.diff(np.r_[0.0, yaws]) / float(FUTURE_TIMES_S[0])
                lateral_accel = float(np.max(np.abs(intervals * yaw_rate)))
                return road_fraction(hdmap, positions, yaws), lateral_accel
            gt_road, follower_lateral = sampled_road(arm, action)
            raw_gt_road, raw_follower_lateral = sampled_road(winner, 0)
            for name, value in (
                ("curvature_1_per_m", curvature),
                ("raw_curvature_1_per_m", raw_curvature),
                ("corridor_point_max", point_cost),
                ("raw_corridor_point_max", raw_point_cost),
                ("lateral_accel_proxy_mps2", lateral_accel),
                ("track_error_m", error),
                ("raw_track_error_m", raw_error),
                ("progress_m", progress),
                ("gt_road_fraction", gt_road),
                ("raw_gt_road_fraction", raw_gt_road),
                ("follower_lateral_accel_mps2", follower_lateral),
                ("raw_follower_lateral_accel_mps2", raw_follower_lateral),
            ):
                measurements[name].append(value)
            details.append({
                "key": str(base["key"][row]), "arm": arm, "action": action,
                "target_speed_mps": speed, "progress_m": progress,
                "point_corridor_max": point_cost, "curvature_1_per_m": curvature,
                "lateral_accel_proxy_mps2": lateral_accel,
                "track_error_m": error, "raw_track_error_m": raw_error,
                "gt_road_fraction": gt_road,
                "raw_gt_road_fraction": raw_gt_road,
                "follower_lateral_accel_mps2": follower_lateral,
                "raw_follower_lateral_accel_mps2": raw_follower_lateral,
            })
        values = {key: np.asarray(value) for key, value in measurements.items()}
        results[prefix] = {
            "incremental_rescues": len(rows),
            "distinct_routes": len({str(base["key"][row]).rsplit("__", 1)[0]
                                    for row in rows}),
            "unchanged_target_count": int((expanded[rows, 1] == 0).sum()),
            "raw_track_error_over_1m_count": int(
                (values["raw_track_error_m"] > 1.0).sum()
            ),
            "selected_point_corridor_over_0p1_count": int(
                (values["corridor_point_max"] > 0.1).sum()
            ),
            "selected_point_corridor_over_0p25_count": int(
                (values["corridor_point_max"] > 0.25).sum()
            ),
            "curvature_over_0p2_count": int((values["curvature_1_per_m"] > 0.2).sum()),
            "lateral_accel_proxy_over_3_count": int(
                (values["lateral_accel_proxy_mps2"] > 3.0).sum()
            ),
            "gt_road_fraction_below_0p95_count": int(
                (values["gt_road_fraction"] < 0.95).sum()
            ),
            "raw_gt_road_fraction_below_0p95_count": int(
                (values["raw_gt_road_fraction"] < 0.95).sum()
            ),
            "selected_road_fraction_lower_than_raw_count": int(
                (values["gt_road_fraction"] + 0.01 < values["raw_gt_road_fraction"]).sum()
            ),
            "follower_lateral_accel_over_3_count": int(
                (values["follower_lateral_accel_mps2"] > 3.0).sum()
            ),
            "raw_follower_lateral_accel_over_3_count": int(
                (values["raw_follower_lateral_accel_mps2"] > 3.0).sum()
            ),
            "follower_lateral_accel_over_raw_plus_1_count": int((
                values["follower_lateral_accel_mps2"]
                > values["raw_follower_lateral_accel_mps2"] + 1.0
            ).sum()),
            "road_and_relative_accel_pass_count": int((
                (values["gt_road_fraction"] >= 0.95)
                & (values["follower_lateral_accel_mps2"]
                   <= values["raw_follower_lateral_accel_mps2"] + 1.0)
            ).sum()),
            "summary": {name: quantiles(value) for name, value in values.items()},
            "frames": details,
        }
    result = {
        "joint_report": str(joint_report.resolve()),
        "pid_report": str(pid_report.resolve()),
        "hdmap_root": str(hdmap_root.resolve()),
        "limitation": "GT static HD-map swept narrow-ego footprint and predicted corridor; no dynamic road occupancy, CARLA tire model or closed-loop tracking proof.",
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-report", type=Path, default=root / "outputs/local_training/p5_stepB3b_joint_oracle/joint_nested_k16_k32_local_8gpu.json")
    parser.add_argument("--pid-report", type=Path, default=root / "outputs/local_training/p5_stepB3b_joint_oracle/pid_joint_coverage_5000.json")
    parser.add_argument("--out", type=Path, default=root / "outputs/local_training/p5_stepB3b_joint_oracle/pid_local_feasibility_5000.json")
    parser.add_argument("--hdmap-root", type=Path, default=root / "data/carla_leaderboard2/data")
    args = parser.parse_args()
    result = audit(args.joint_report, args.pid_report, args.out, args.hdmap_root)
    for name, metrics in result["results"].items():
        print(f"{name}: incremental={metrics['incremental_rescues']} "
              f"point-corridor>0.1={metrics['selected_point_corridor_over_0p1_count']} "
              f"curvature>0.2={metrics['curvature_over_0p2_count']} "
              f"a_lat>3={metrics['lateral_accel_proxy_over_3_count']} "
              f"follower-a_lat>3={metrics['follower_lateral_accel_over_3_count']} "
              f"GT-road<95%={metrics['gt_road_fraction_below_0p95_count']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
