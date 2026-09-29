"""No-training oracle for executable scalar-speed actions on the frozen Path.

The longitudinal controller is exact; vehicle acceleration and fixed-Path
tracking are proxies. GT future actors give counterfactual collision labels.
This is a feasibility test, NOT a deployable selector or a closed-loop score.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lead.data_loader.future_actor_cache import (
    FUTURE_TIMES_S,
    unpack_future_actor_frame,
)
from lead.expert.config_expert import ExpertConfig
from lead.tfv6.control_aware_velocity_oracle import (
    ACTION_NAMES,
    Dynamics,
    rollout_scalar_target,
    scalar_actions,
)
from lead.tfv6.future_collision import (
    future_collision_label,
    interpolate_route_by_distance,
)
from scripts.p4.audit_b3a_control_velocity_oracle import (
    actor_locations,
    interpolate_samples,
    reconstruct_route_samples,
)


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _route_arc_length(route: np.ndarray) -> float:
    route = np.asarray(route, dtype=np.float32).reshape(-1, 2)
    if len(route) == 0:
        return 0.0
    if np.linalg.norm(route[0]) > 1e-4:
        route = np.concatenate((np.zeros((1, 2), dtype=np.float32), route))
    return float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())


def evaluate_frame(
    *,
    current_speed: float,
    raw_target: float,
    candidate_velocity: np.ndarray,
    candidate_states: np.ndarray | None,
    candidate_valid: np.ndarray,
    selected_path: np.ndarray | None = None,
    actors,
    dynamics: Dynamics,
    expert_config: ExpertConfig,
    margin_m: float,
    max_progress_loss_m: float,
) -> dict:
    """Evaluate all actions on one confidence-selected geometric Path."""
    if not candidate_valid[0]:
        raise ValueError("raw fixed-Path candidate is invalid")
    samples = None
    if selected_path is None:
        if candidate_states is None:
            raise ValueError("candidate_states are required without selected_path")
        samples = reconstruct_route_samples(
            candidate_velocity.astype(np.float32), candidate_states, candidate_valid
        )
    targets, valid = scalar_actions(raw_target)
    count = len(targets)
    collision = np.zeros(count, dtype=bool)
    ttc = np.full(count, float(FUTURE_TIMES_S[-1]), dtype=np.float32)
    progress = np.zeros(count, dtype=np.float32)
    comfort = np.zeros(count, dtype=np.float32)
    brake_fraction = np.zeros(count, dtype=np.float32)
    terminal_speed = np.zeros(count, dtype=np.float32)
    route_overflow = np.zeros(count, dtype=bool)
    route_end_m = (
        _route_arc_length(selected_path)
        if selected_path is not None
        else float(samples[0][-1])
    )
    for index in np.flatnonzero(valid):
        rollout = rollout_scalar_target(
            current_speed,
            float(targets[index]),
            dynamics=dynamics,
            expert_config=expert_config,
        )
        xy, yaw = (
            interpolate_route_by_distance(
                selected_path, rollout.xy_distance_m, extrapolate=True
            )
            if selected_path is not None
            else interpolate_samples(samples, rollout.xy_distance_m)
        )
        label = future_collision_label(
            xy,
            yaw,
            actors,
            safety_margin_m=margin_m,
            include_class_ids=(1, 2),
        )
        collision[index] = label.collision
        ttc[index] = min(float(label.ttc_s), float(FUTURE_TIMES_S[-1]))
        progress[index] = rollout.progress_m
        comfort[index] = rollout.comfort
        brake_fraction[index] = rollout.brake_ticks / len(rollout.acceleration_mps2)
        terminal_speed[index] = rollout.speed_mps[-1]
        route_overflow[index] = rollout.progress_m > route_end_m + 0.01
    safe = valid & ~collision
    slow_only = targets <= targets[0] + 1e-3
    reasonable_progress = progress >= progress[0] - max_progress_loss_m
    # Oracle chooses only when the baseline collides. Safety first, then progress,
    # then comfort. No GT future is used by an inference-time policy here.
    eligible = safe & slow_only & reasonable_progress
    selected = 0
    if collision[0] and eligible.any():
        indices = np.flatnonzero(eligible)
        selected = int(max(indices, key=lambda i: (progress[i], comfort[i])))
    return {
        "targets_mps": targets,
        "valid": valid,
        "collision": collision,
        "ttc_s": ttc,
        "progress_m": progress,
        "comfort": comfort,
        "brake_fraction": brake_fraction,
        "terminal_speed_mps": terminal_speed,
        "route_overflow": route_overflow,
        "safe_any": bool(safe.any()),
        "safe_slow": bool((safe & slow_only).any()),
        "safe_eligible": bool(eligible.any()),
        "selected_index": selected,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=root
        / "outputs/local_training/p5_stepB3a_v2_scene_scorer/features/heldout",
    )
    parser.add_argument(
        "--future-actors",
        type=Path,
        default=root
        / "outputs/local_training/p5_stepB3a_v2_dense_data/future_actor_cache/heldout_future_actors",
    )
    parser.add_argument(
        "--selected-paths",
        type=Path,
        help="directory of aligned selected_paths_*.npz sidecars; use full Paths",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "outputs/local_training/p5_stepB3a_control_aware_oracle/control_aware_oracle.json",
    )
    parser.add_argument(
        "--limit", type=int, default=5000, help="0 evaluates all chosen frames"
    )
    parser.add_argument(
        "--sample-per-shard", type=int, default=600, help="0 selects every frame"
    )
    parser.add_argument("--drive-accel", type=float, default=4.8)
    parser.add_argument("--brake-decel", type=float, default=6.0)
    parser.add_argument("--drag", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--max-progress-loss", type=float, default=2.0)
    args = parser.parse_args()
    if args.limit < 0 or args.sample_per_shard < 0 or args.max_progress_loss < 0:
        parser.error(
            "limit, sample-per-shard and max-progress-loss must be nonnegative"
        )
    if args.drive_accel <= 0 or args.brake_decel <= 0 or args.drag < 0:
        parser.error(
            "dynamics must have positive acceleration/deceleration and nonnegative drag"
        )
    dynamics = Dynamics(args.drive_accel, args.brake_decel, args.drag)
    expert_config = ExpertConfig()
    feature_paths = sorted(args.features.glob("velocity_features_*.npz"))
    if not feature_paths:
        raise FileNotFoundError(f"no feature shards in {args.features}")
    locations = actor_locations(args.future_actors)
    loaded: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
    records: list[dict] = []
    keys: list[str] = []
    counts = Counter()
    selected_actions = Counter()
    for path in feature_paths:
        with np.load(path, allow_pickle=False) as feature_file:
            required = (
                "keys",
                "current_speed",
                "raw_target_speed",
                "candidate_velocity",
                "candidate_valid",
                "selected_arm",
            )
            if args.selected_paths is None:
                required += ("candidate_states",)
            missing = [name for name in required if name not in feature_file]
            if missing:
                raise ValueError(f"{path} missing {missing}")
            arrays = {name: feature_file[name] for name in required}
            feature_checkpoint = str(feature_file["source_checkpoint"])
            feature_manifest = str(feature_file["source_manifest"])
            feature_vlm_manifest = str(feature_file["nearest_vlm_manifest"])
            feature_pair_id = (
                str(feature_file["extraction_pair_id"])
                if "extraction_pair_id" in feature_file
                else None
            )
        selected_paths = None
        if args.selected_paths is not None:
            path_file = args.selected_paths / path.name.replace(
                "velocity_features_", "selected_paths_"
            )
            if not path_file.is_file():
                raise FileNotFoundError(
                    f"missing aligned selected Path shard: {path_file}"
                )
            with np.load(path_file, allow_pickle=False) as sidecar:
                sidecar_pair_id = (
                    str(sidecar["extraction_pair_id"])
                    if "extraction_pair_id" in sidecar
                    else None
                )
                if feature_pair_id != sidecar_pair_id:
                    raise ValueError(f"selected Path pair ID mismatch: {path_file}")
                if not np.array_equal(sidecar["keys"], arrays["keys"]):
                    raise ValueError(f"selected Path keys/order mismatch: {path_file}")
                if str(sidecar["source_checkpoint"]) != feature_checkpoint:
                    raise ValueError(f"selected Path checkpoint mismatch: {path_file}")
                if str(sidecar["source_manifest"]) != feature_manifest:
                    raise ValueError(f"selected Path manifest mismatch: {path_file}")
                if str(sidecar["nearest_vlm_manifest"]) != feature_vlm_manifest:
                    raise ValueError(
                        f"selected Path VLM manifest mismatch: {path_file}"
                    )
                if not np.array_equal(sidecar["selected_arm"], arrays["selected_arm"]):
                    raise ValueError(f"selected Path arm mismatch: {path_file}")
                for name in ("current_speed", "raw_target_speed"):
                    if not np.allclose(
                        sidecar[name], arrays[name].astype(np.float32), atol=0.03
                    ):
                        raise ValueError(f"selected Path {name} mismatch: {path_file}")
                selected_paths = sidecar["selected_path"].astype(np.float32)
                if (
                    selected_paths.ndim != 3
                    or selected_paths.shape[0] != len(arrays["keys"])
                    or selected_paths.shape[2] != 2
                    or not np.isfinite(selected_paths).all()
                ):
                    raise ValueError(f"invalid selected Path shape/values: {path_file}")
        n = len(arrays["keys"])
        rows = (
            np.unique(np.linspace(0, n - 1, min(n, args.sample_per_shard), dtype=int))
            if args.sample_per_shard
            else range(n)
        )
        for row in tqdm(rows, desc=path.stem, leave=False):
            if args.limit and len(records) >= args.limit:
                break
            key = str(arrays["keys"][row])
            if key not in locations:
                raise KeyError(f"missing future actors for {key}")
            actor_path, actor_index = locations[key]
            if actor_path not in loaded:
                with np.load(actor_path, allow_pickle=False) as actor_file:
                    loaded[actor_path] = {
                        name: actor_file[name] for name in actor_file.files
                    }
                if len(loaded) > 2:
                    loaded.popitem(last=False)
            actors = unpack_future_actor_frame(loaded[actor_path], actor_index)
            record = evaluate_frame(
                current_speed=float(arrays["current_speed"][row]),
                raw_target=float(arrays["raw_target_speed"][row]),
                candidate_velocity=arrays["candidate_velocity"][row],
                candidate_states=(
                    None
                    if args.selected_paths is not None
                    else arrays["candidate_states"][row]
                ),
                candidate_valid=arrays["candidate_valid"][row],
                selected_path=(None if selected_paths is None else selected_paths[row]),
                actors=actors,
                dynamics=dynamics,
                expert_config=expert_config,
                margin_m=args.margin,
                max_progress_loss_m=args.max_progress_loss,
            )
            keys.append(key)
            records.append(record)
            raw_collision = bool(record["collision"][0])
            selected = record["selected_index"]
            counts["raw_collision"] += raw_collision
            counts["oracle_all_collision"] += raw_collision and not record["safe_any"]
            counts["oracle_slow_collision"] += raw_collision and not record["safe_slow"]
            counts["oracle_progress_constrained_collision"] += (
                raw_collision and not record["safe_eligible"]
            )
            counts["rescued_with_progress_bound"] += raw_collision and selected != 0
            counts["rescued_only_by_stop"] += (
                raw_collision
                and record["safe_slow"]
                and not np.any(
                    record["valid"]
                    & ~record["collision"]
                    & (record["terminal_speed_mps"] > 0.5)
                )
            )
            counts["baseline_route_overflow"] += bool(record["route_overflow"][0])
            selected_actions[ACTION_NAMES[selected]] += 1
        if args.limit and len(records) >= args.limit:
            break
    if not records:
        raise ValueError("no held-out frames evaluated")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output.with_suffix(".frames.npz")
    np.savez_compressed(
        npz_path,
        keys=np.asarray(keys),
        action_names=np.asarray(ACTION_NAMES),
        **{
            field: np.stack([record[field] for record in records])
            for field in (
                "targets_mps",
                "valid",
                "collision",
                "ttc_s",
                "progress_m",
                "comfort",
                "brake_fraction",
                "terminal_speed_mps",
                "route_overflow",
            )
        },
        selected_index=np.asarray(
            [record["selected_index"] for record in records], dtype=np.int8
        ),
    )
    rescued = [record for record in records if record["selected_index"] != 0]
    # The cached feature shard contains candidate positions, not the complete
    # original Path. Extrapolation beyond the farthest sampled point is useful
    # for a sensitivity check, but must not count as verified geometry.
    geometry_qualified = [
        record for record in records if not record["route_overflow"][0]
    ]
    raw_collision_qualified = sum(
        bool(record["collision"][0]) for record in geometry_qualified
    )
    rescue_curve = {}
    for loss_bound in (1.0, 2.0, 4.0, 8.0, float("inf")):
        rescued_count = 0
        for record in geometry_qualified:
            if not record["collision"][0]:
                continue
            eligible = (
                record["valid"]
                & ~record["collision"]
                & ~record["route_overflow"]
                & (record["targets_mps"] <= record["targets_mps"][0] + 1e-3)
                & (record["progress_m"] >= record["progress_m"][0] - loss_bound)
            )
            rescued_count += bool(eligible.any())
        rescue_curve[str(loss_bound)] = {
            "rescued_frames": rescued_count,
            "rescue_given_raw_collision": _rate(rescued_count, raw_collision_qualified),
        }
    report = {
        "scope": "fixed confidence-selected Path; scalar target actions; exact longitudinal decision logic; approximate vehicle acceleration; GT future actors",
        "closed_loop_qualified": False,
        "frames": len(records),
        "action_names": list(ACTION_NAMES),
        "dynamics": vars(dynamics),
        "collision_margin_m": args.margin,
        "max_progress_loss_m": args.max_progress_loss,
        "counts": dict(counts),
        "rates": {key: _rate(value, len(records)) for key, value in counts.items()},
        "rescue_given_raw_collision": _rate(
            counts["rescued_with_progress_bound"], counts["raw_collision"]
        ),
        "selected_action_counts": dict(selected_actions),
        "rescued_progress_loss_m_mean": float(
            np.mean(
                [
                    record["progress_m"][0]
                    - record["progress_m"][record["selected_index"]]
                    for record in rescued
                ]
            )
        )
        if rescued
        else 0.0,
        "rescued_extra_brake_fraction_mean": float(
            np.mean(
                [
                    record["brake_fraction"][record["selected_index"]]
                    - record["brake_fraction"][0]
                    for record in rescued
                ]
            )
        )
        if rescued
        else 0.0,
        "geometry_qualified": {
            "frames": len(geometry_qualified),
            "raw_collision_frames": raw_collision_qualified,
            "slowdown_rescue_by_max_progress_loss_m": rescue_curve,
        },
        "features": str(args.features.resolve()),
        "selected_paths": (
            None if args.selected_paths is None else str(args.selected_paths.resolve())
        ),
        "future_actors": str(args.future_actors.resolve()),
        "frame_labels": str(npz_path.resolve()),
        "limitations": "Counterfactual GT actors, fixed Path and fixed scalar target for 2 s. No CARLA physics, lateral tracking error, perception error or receding-horizon replanning. Do not deploy oracle decisions.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {output} and {npz_path}")


if __name__ == "__main__":
    main()
