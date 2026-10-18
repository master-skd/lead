#!/usr/bin/env python3
"""Summarize opt-in P6 traces; all quality signals are diagnostic proxies."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def summarize(records: list[dict]) -> dict:
    """Compare available candidates with selected candidate, then execution."""
    if not records:
        raise ValueError("empty P6 trace")
    selected_lane, best_lane, lane_misses = [], [], []
    nav_selected, nav_best = [], []
    low_speed_while_stopped = brake_while_stopped = 0
    stopped = 0
    forecast_errors = []
    for i, rec in enumerate(records):
        candidate = rec["candidates"]
        selection = rec["selection"]
        selected = selection["index"]
        valid = rec["intent"]["valid"]
        fractions = candidate["driving_lane_fraction"]
        if fractions is not None:
            best = max(f for f, is_valid in zip(fractions, valid, strict=True) if is_valid)
            picked = fractions[selected]
            selected_lane.append(picked)
            best_lane.append(best)
            if best >= 0.875 and picked < 0.75:
                lane_misses.append((rec["step"], round(best, 3), round(picked, 3)))
        target = rec["navigation"]["target_point_ego_m"]
        if target is not None and math.hypot(*target) > 2.0:
            nav_angle = math.atan2(target[1], target[0])
            errors = []
            for route in candidate["routes_ego_m"]:
                end = route[-1]
                angle = math.atan2(end[1], end[0])
                errors.append(abs(math.atan2(math.sin(angle-nav_angle), math.cos(angle-nav_angle))))
            nav_selected.append(math.degrees(errors[selected]))
            nav_best.append(math.degrees(min(err for err, ok in zip(errors, valid, strict=True) if ok)))
        if rec["current_speed_mps"] < 0.5:
            stopped += 1
            low_speed_while_stopped += selection["executed_target_speed_mps"] < 0.5
            brake_while_stopped += rec["controller"]["commanded"]["brake"] > 0.5

        states = candidate["states_ego_2s"]
        if states is None:
            continue
        t = rec["game_time_s"] + 1.0
        # Records are ordered by simulator time. With optional subsampling,
        # accept the first position within 0.15 s of the one-second horizon.
        j = i + 1
        while j < len(records) and records[j]["game_time_s"] < t:
            j += 1
        if j == len(records) or records[j]["game_time_s"] - t > 0.15:
            continue
        origin, future = rec["ego_world"], records[j]["ego_world"]
        dx, dy = future["x"] - origin["x"], future["y"] - origin["y"]
        yaw = math.radians(origin["yaw_deg"])
        actual_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        actual_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        pred_x, pred_y = states[selected][3][:2]  # 8 x 0.25 s states, index 3 = 1 s
        forecast_errors.append(math.hypot(actual_x-pred_x, actual_y-pred_y))
    return {
        "frames": len(records),
        "mean_valid_arms": _mean([r["selection"]["valid_count"] for r in records]),
        "map_sampled_frames": len(selected_lane),
        "selected_driving_lane_fraction": _mean(selected_lane),
        "best_valid_driving_lane_fraction": _mean(best_lane),
        "map_lane_miss_frames": lane_misses,
        "selected_nav_angle_error_deg": _mean(nav_selected),
        "best_valid_nav_angle_error_deg": _mean(nav_best),
        "stopped_frames": stopped,
        "stopped_low_target_speed_fraction": low_speed_while_stopped / stopped if stopped else None,
        "stopped_braking_fraction": brake_while_stopped / stopped if stopped else None,
        "mean_selected_1s_execution_error_m": _mean(forecast_errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_root", type=Path)
    args = parser.parse_args()
    traces = sorted(args.evaluation_root.glob("*/p6_candidate_trace.jsonl"))
    if not traces:
        raise SystemExit(f"No P6 traces found under {args.evaluation_root}")
    for path in traces:
        with path.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        result = summarize(records)
        checkpoint = path.parent / "checkpoint_endpoint.json"
        status = "no checkpoint record"
        if checkpoint.is_file():
            with checkpoint.open(encoding="utf-8") as stream:
                route_records = json.load(stream).get("_checkpoint", {}).get("records", [])
            if route_records:
                status = route_records[0].get("status", status)
        print(f"\nRoute {path.parent.name}: {status}; {result['frames']} frames, "
              f"{result['map_sampled_frames']} CARLA-map samples")
        print(f"  valid arms mean={_fmt(result['mean_valid_arms'])}; "
              f"driving-lane fraction selected/best={_fmt(result['selected_driving_lane_fraction'])}/"
              f"{_fmt(result['best_valid_driving_lane_fraction'])}; "
              f"clear map-lane misses={len(result['map_lane_miss_frames'])}")
        print(f"  nav angle error selected/best={_fmt(result['selected_nav_angle_error_deg'])}/"
              f"{_fmt(result['best_valid_nav_angle_error_deg'])} deg (coarse direction proxy)")
        print(f"  stopped frames={result['stopped_frames']}; low target-speed fraction="
              f"{_fmt(result['stopped_low_target_speed_fraction'])}; braking fraction="
              f"{_fmt(result['stopped_braking_fraction'])}")
        print(f"  1 s selected-path vs actual-position error="
              f"{_fmt(result['mean_selected_1s_execution_error_m'])} m "
              "(replanning/control diagnostic)")
        if result["map_lane_miss_frames"]:
            print("  first map-lane miss (step, best, selected): "
                  f"{result['map_lane_miss_frames'][:8]}")


if __name__ == "__main__":
    main()
