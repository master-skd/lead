"""Audit B3a velocity coverage after profiles collapse to PID target speeds.

This is an offline GT-actor oracle, not a CARLA/PID rollout. The execution proxy
travels at the candidate's first-interval mean speed for 0.25 s, then holds the
target sent to PID. Geometry is reconstructed from cached fixed-route candidates;
the reconstruction error is reported so this approximation cannot silently pass
as an exact collision label.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lead.data_loader.future_actor_cache import FUTURE_TIMES_S, unpack_future_actor_frame
from lead.tfv6.future_collision import future_collision_label


def control_target(current_speed: float, first_interval_speed: float) -> float:
    """The exact scalar mapping used by the closed-loop velocity selector."""
    return max(0.0, 2.0 * first_interval_speed - current_speed)


def hold_target_distances(first_speed: float, target: float) -> np.ndarray:
    """A control-conditioned, constant-target *proxy*, not a PID simulation."""
    dt = float(FUTURE_TIMES_S[0])
    intervals = np.full(len(FUTURE_TIMES_S), target, dtype=np.float32)
    intervals[0] = first_speed
    return np.cumsum(intervals) * dt


def reconstruct_route_samples(
    velocities: np.ndarray, states: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool all fixed-route profile samples, indexed by traveled arc distance."""
    dt = float(FUTURE_TIMES_S[0])
    distances = (np.cumsum(velocities[valid].astype(np.float32), axis=1) * dt).ravel()
    samples = states[valid].astype(np.float32).reshape(-1, 6)
    # Quantization groups nearly identical arc samples in fp16 caches.
    groups = np.round(distances, 3)
    order = np.argsort(groups, kind="stable")
    groups, samples = groups[order], samples[order]
    unique, starts, counts = np.unique(groups, return_index=True, return_counts=True)
    xy = np.stack([
        np.add.reduceat(samples[:, axis], starts) / counts for axis in (0, 1)
    ], axis=1)
    yaw = np.arctan2(
        np.add.reduceat(samples[:, 2], starts) / counts,
        np.add.reduceat(samples[:, 3], starts) / counts,
    )
    if len(unique) == 0 or unique[0] > 1e-3:
        unique = np.concatenate(([0.0], unique))
        xy = np.concatenate((np.zeros((1, 2), dtype=np.float32), xy))
        yaw = np.concatenate(([float(yaw[0]) if len(yaw) else 0.0], yaw))
    return unique, xy, yaw


def interpolate_samples(
    samples: tuple[np.ndarray, np.ndarray, np.ndarray], distances: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    arc, xy, yaw = samples
    distance = np.asarray(distances, dtype=np.float32)
    positions = np.stack([
        np.interp(distance, arc, xy[:, axis]) for axis in (0, 1)
    ], axis=1).astype(np.float32)
    # The original interpolator uses piecewise-constant route-segment yaw. Nearest
    # cached yaw preserves that property better than interpolating across a turn.
    right = np.searchsorted(arc, distance)
    right = np.clip(right, 0, len(arc) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(abs(arc[right] - distance) < abs(arc[left] - distance), right, left)
    headings = yaw[nearest].astype(np.float32)
    beyond = distance > arc[-1]
    if beyond.any() and len(arc) > 1:
        tangent = xy[-1] - xy[-2]
        tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
        positions[beyond] = xy[-1] + (distance[beyond] - arc[-1])[:, None] * tangent
        headings[beyond] = np.arctan2(tangent[1], tangent[0])
    return positions, headings


def actor_locations(cache_dir: Path) -> dict[str, tuple[Path, int]]:
    locations = {}
    for path in sorted(cache_dir.glob("future_actors_*.npz")):
        with np.load(path, allow_pickle=False) as shard:
            for index, key in enumerate(shard["keys"]):
                if str(key) in locations:
                    raise ValueError(f"duplicate actor key: {key}")
                locations[str(key)] = path, index
    if not locations:
        raise FileNotFoundError(f"no actor shards in {cache_dir}")
    return locations


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=root / "outputs/local_training/p5_stepB3a_v2_scene_scorer/features/heldout")
    parser.add_argument("--future-actors", type=Path, default=root / "outputs/local_training/p5_stepB3a_v2_dense_data/future_actor_cache/heldout_future_actors")
    parser.add_argument("--output", type=Path, default=root / "outputs/local_training/p5_stepB3a_control_oracle/control_oracle.json")
    parser.add_argument("--limit", type=int, default=5000, help="0 evaluates all held-out frames")
    parser.add_argument("--sample-per-shard", type=int, default=0, help="evenly spaced rows per feature shard; 0 selects consecutive rows")
    parser.add_argument("--action-precision", type=int, default=2, help="decimal places for equivalent PID targets")
    args = parser.parse_args()
    feature_paths = sorted(args.features.glob("velocity_features_*.npz"))
    if not feature_paths:
        raise FileNotFoundError(f"no feature shards in {args.features}")
    locations = actor_locations(args.future_actors)
    loaded: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
    counts = {key: 0 for key in (
        "frames", "raw_full_collision", "oracle_full_collision", "raw_proxy_collision",
        "oracle_proxy_collision", "proxy_rescue", "proxy_all_unsafe", "proxy_moving_rescue",
        "duplicate_actions", "full_label_conflict",
        "raw_label_mismatch", "positive_geometry_error", "geometry_points",
    )}
    sums = {"valid_candidates": 0.0, "unique_actions": 0.0, "geometry_abs_m": 0.0}
    maximum_geometry_error = 0.0
    processed = 0
    for path in feature_paths:
        with np.load(path, allow_pickle=False) as features:
            required = ("keys", "candidate_velocity", "candidate_valid", "candidate_states", "collision", "current_speed")
            missing = [name for name in required if name not in features]
            if missing:
                raise ValueError(f"{path} lacks scene-v2 data: {missing}")
            # NpzFile decompresses an entire member on every indexing operation.
            # Materialize the six fields once per shard, then index in memory.
            arrays = {name: features[name] for name in required}
            n = len(arrays["keys"])
            if args.sample_per_shard:
                chosen_rows = np.unique(np.linspace(
                    0, n - 1, min(args.sample_per_shard, n), dtype=int
                ))
            else:
                remaining = n if not args.limit else min(n, args.limit - processed)
                chosen_rows = range(remaining)
            for row in tqdm(chosen_rows, desc=path.stem, leave=False):
                if args.limit and processed >= args.limit:
                    break
                key = str(arrays["keys"][row])
                if key not in locations:
                    raise KeyError(f"missing future actors for {key}")
                actor_path, actor_index = locations[key]
                if actor_path not in loaded:
                    with np.load(actor_path, allow_pickle=False) as shard:
                        loaded[actor_path] = {name: shard[name] for name in shard.files}
                    if len(loaded) > 2:
                        loaded.popitem(last=False)
                actors = unpack_future_actor_frame(loaded[actor_path], actor_index)
                valid = arrays["candidate_valid"][row].astype(bool)
                velocities = arrays["candidate_velocity"][row].astype(np.float32)
                states = arrays["candidate_states"][row]
                full_labels = arrays["collision"][row].astype(bool)
                current = float(arrays["current_speed"][row])
                if not valid[0]:
                    raise ValueError(f"raw candidate invalid for {key}")
                samples = reconstruct_route_samples(velocities, states, valid)
                candidate_indices = np.flatnonzero(valid)
                cached_xy = states[valid, :, :2].astype(np.float32).reshape(-1, 2)
                cached_dist = (np.cumsum(velocities[valid], axis=1) * float(FUTURE_TIMES_S[0])).ravel()
                restored_xy, _ = interpolate_samples(samples, cached_dist)
                errors = np.linalg.norm(restored_xy - cached_xy, axis=1)
                sums["geometry_abs_m"] += float(errors.sum())
                counts["geometry_points"] += len(errors)
                counts["positive_geometry_error"] += int((errors > 0.1).sum())
                maximum_geometry_error = max(maximum_geometry_error, float(errors.max()))

                groups: dict[float, list[int]] = {}
                for index in candidate_indices:
                    target = control_target(current, float(velocities[index, 0]))
                    groups.setdefault(round(target, args.action_precision), []).append(int(index))
                proxy = np.zeros(len(valid), dtype=bool)
                for indices in groups.values():
                    index = indices[0]
                    first = float(velocities[index, 0])
                    target = control_target(current, first)
                    distances = hold_target_distances(first, target)
                    xy, yaw = interpolate_samples(samples, distances)
                    label = future_collision_label(
                        xy, yaw, actors, safety_margin_m=0.2, include_class_ids=(1, 2)
                    ).collision
                    proxy[indices] = label
                    if len(indices) > 1 and np.unique(full_labels[indices]).size > 1:
                        counts["full_label_conflict"] += 1
                raw_full = bool(full_labels[0])
                raw_proxy = bool(proxy[0])
                any_full_safe = bool((~full_labels[valid]).any())
                any_proxy_safe = bool((~proxy[valid]).any())
                counts["frames"] += 1
                counts["raw_full_collision"] += raw_full
                counts["oracle_full_collision"] += not any_full_safe
                counts["raw_proxy_collision"] += raw_proxy
                counts["oracle_proxy_collision"] += not any_proxy_safe
                counts["proxy_all_unsafe"] += not any_proxy_safe
                counts["proxy_rescue"] += raw_proxy and any_proxy_safe
                counts["proxy_moving_rescue"] += raw_proxy and any(
                    not proxy[i] and velocities[i, 0] > 0.5 for i in candidate_indices
                )
                counts["duplicate_actions"] += len(groups) < len(candidate_indices)
                counts["raw_label_mismatch"] += raw_full != raw_proxy
                sums["valid_candidates"] += len(candidate_indices)
                sums["unique_actions"] += len(groups)
                processed += 1
        if args.limit and processed >= args.limit:
            break
    n = counts["frames"]
    report = {
        "scope": "fixed confidence-selected path; GT future actors; 2 s constant-PID-target proxy",
        "reference": "candidate zero after velocity-selector PID mapping, not the unmodified corridor baseline controller",
        "frames": n,
        "feature_dir": str(args.features.resolve()),
        "actor_dir": str(args.future_actors.resolve()),
        "action_precision_decimals": args.action_precision,
        "sample_per_shard": args.sample_per_shard,
        "valid_candidates_per_frame": sums["valid_candidates"] / n,
        "unique_pid_targets_per_frame": sums["unique_actions"] / n,
        "fraction_with_duplicate_targets": counts["duplicate_actions"] / n,
        "full_profile_collision": {
            "raw": counts["raw_full_collision"] / n,
            "oracle_floor": counts["oracle_full_collision"] / n,
        },
        "control_proxy_collision": {
            "raw": counts["raw_proxy_collision"] / n,
            "oracle_floor": counts["oracle_proxy_collision"] / n,
            "raw_collision_frames": counts["raw_proxy_collision"],
            "rescuable_frames": counts["proxy_rescue"],
            "moving_rescuable_frames": counts["proxy_moving_rescue"],
            "rescue_given_raw_collision": counts["proxy_rescue"] / max(counts["raw_proxy_collision"], 1),
            "moving_rescue_given_raw_collision": counts["proxy_moving_rescue"] / max(counts["raw_proxy_collision"], 1),
        },
        "full_profile_vs_proxy_raw_disagreement": counts["raw_label_mismatch"] / n,
        "action_groups_with_conflicting_full_profile_labels": counts["full_label_conflict"],
        "geometry_reconstruction": {
            "mean_point_error_m": sums["geometry_abs_m"] / max(counts["geometry_points"], 1),
            "fraction_points_over_0.1m": counts["positive_geometry_error"] / max(counts["geometry_points"], 1),
            "max_point_error_m": maximum_geometry_error,
        },
        "limitations": "GT future actors and constant target hold; not the real CARLA PID or receding closed loop. Route reconstructed from fp16 candidate states.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
