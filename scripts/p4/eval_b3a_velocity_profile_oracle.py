"""Expanded no-training velocity-profile oracle on a fixed spatial route.

Unlike the target-speed-only B3a-0 audit, this audit represents candidates as
complete two-second velocity sequences.  It adds explicit braking/stopping
profiles and evaluates several collision margins in one model pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_BATCH_KEYS = {
    "rgb",
    "rasterized_lidar",
    "vlm_hidden",
    "radar",
    "speed",
    "command",
    "target_point",
    "target_point_previous",
    "target_point_next",
    "route",
    "route_number",
    "frame_number",
    "anchor",
    "target_speed",
    "brake",
}


@dataclass(frozen=True)
class VelocityProfile:
    name: str
    family: str
    velocities: np.ndarray
    distances: np.ndarray
    target_speed: float


def _collate(batch: list[dict]) -> dict:
    slim = [
        {key: value for key, value in sample.items() if key in _BATCH_KEYS}
        for sample in batch
    ]
    return torch.utils.data._utils.collate.default_collate(slim)


def _parse_float_tuple(
    value: str, *, minimum: float | None = None
) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("expected at least one comma-separated value")
    if minimum is not None and any(item < minimum for item in result):
        raise ValueError(f"values must be at least {minimum}")
    return result


def _target_profile(
    current_speed: float,
    target_speed: float,
    times_s: np.ndarray,
    speed_profile_distances,
    *,
    name: str,
) -> VelocityProfile:
    v0 = max(0.0, float(current_speed))
    target = max(0.0, float(target_speed))
    horizon = max(float(times_s[-1]), 1e-6)
    acceleration = float(np.clip((target - v0) / horizon, -4.95, 1.89))
    if abs(acceleration) < 1e-8:
        velocities = np.full_like(times_s, v0, dtype=np.float32)
    else:
        time_to_target = max(0.0, (target - v0) / acceleration)
        velocities = v0 + acceleration * np.minimum(times_s, time_to_target)
        velocities = np.where(times_s >= time_to_target, target, velocities)
        velocities = np.maximum(velocities, 0.0).astype(np.float32)
    distances = speed_profile_distances(v0, target, times_s=times_s)
    return VelocityProfile(name, "target", velocities, distances, target)


def _acceleration_profile(
    current_speed: float,
    acceleration: float,
    times_s: np.ndarray,
    *,
    name: str,
    family: str,
) -> VelocityProfile:
    """Generate exact distances for constant acceleration followed by a stop."""

    v0 = max(0.0, float(current_speed))
    acceleration = float(np.clip(acceleration, -4.95, 1.89))
    if acceleration < -1e-8:
        stop_time = v0 / -acceleration
        active_time = np.minimum(times_s, stop_time)
        distances = v0 * active_time + 0.5 * acceleration * active_time**2
    else:
        distances = v0 * times_s + 0.5 * acceleration * times_s**2
    velocities = np.maximum(v0 + acceleration * times_s, 0.0)
    return VelocityProfile(
        name,
        family,
        velocities.astype(np.float32),
        np.maximum.accumulate(distances).astype(np.float32),
        float("nan"),
    )


def build_velocity_profiles(
    current_speed: float,
    raw_target_speed: float,
    speed_classes: list[float] | tuple[float, ...],
    fractions: tuple[float, ...],
    stop_times_s: tuple[float, ...],
    times_s: np.ndarray,
    speed_profile_distances,
) -> list[VelocityProfile]:
    """Build unique, physically reachable profiles with no more progress than raw."""

    raw_target = max(0.0, float(raw_target_speed))
    raw = _target_profile(
        current_speed,
        raw_target,
        times_s,
        speed_profile_distances,
        name="target_raw",
    )
    proposals = [raw]
    proposals.extend(
        _target_profile(
            current_speed,
            raw_target * fraction,
            times_s,
            speed_profile_distances,
            name=f"target_fraction_{fraction:g}",
        )
        for fraction in fractions
    )
    proposals.extend(
        _target_profile(
            current_speed,
            speed,
            times_s,
            speed_profile_distances,
            name=f"target_class_{speed:g}",
        )
        for speed in speed_classes
        if 0.0 <= float(speed) <= raw_target + 1e-6
    )
    for name, acceleration in (
        ("brake_emergency", -4.95),
        ("brake_strong", -3.0),
        ("brake_mild", -1.5),
        ("coast", 0.0),
    ):
        proposals.append(
            _acceleration_profile(
                current_speed,
                acceleration,
                times_s,
                name=name,
                family="brake" if acceleration < 0 else "coast",
            )
        )
    for stop_time in stop_times_s:
        acceleration = -max(0.0, float(current_speed)) / max(stop_time, 1e-6)
        proposals.append(
            _acceleration_profile(
                current_speed,
                acceleration,
                times_s,
                name=f"stop_{stop_time:g}s",
                family="stop",
            )
        )

    profiles = []
    raw_progress = float(raw.distances[-1])
    for profile in proposals:
        if profile.name != "target_raw" and profile.distances[-1] > raw_progress + 1e-4:
            continue
        duplicate = any(
            np.allclose(profile.velocities, kept.velocities, atol=1e-5)
            and np.allclose(profile.distances, kept.distances, atol=1e-5)
            for kept in profiles
        )
        if not duplicate:
            profiles.append(profile)
    return profiles


def select_max_progress_safe(collision: np.ndarray, progress: np.ndarray) -> int:
    """Keep raw index zero when safe; otherwise retain maximum safe progress."""

    collision = np.asarray(collision, dtype=bool)
    if not collision[0]:
        return 0
    safe = np.flatnonzero(~collision)
    if len(safe) == 0:
        return 0
    return int(safe[np.argmax(progress[safe])])


def _route_arc_length(route: np.ndarray) -> float:
    route = np.asarray(route, dtype=np.float32).reshape(-1, 2)
    if len(route) == 0:
        return 0.0
    if np.linalg.norm(route[0]) > 1e-4:
        route = np.concatenate([np.zeros((1, 2), dtype=np.float32), route], axis=0)
    return float(np.linalg.norm(np.diff(route, axis=0), axis=-1).sum())


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator / max(denominator, 1))


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"p10": None, "p50": None, "p90": None}
    return {
        "p10": float(np.quantile(values, 0.1)),
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
    }


def summarize_margin(
    arrays: dict[str, np.ndarray], margin_index: int, mask: np.ndarray
) -> dict:
    valid = arrays["candidate_valid"][mask]
    collision = arrays["candidate_collision"][mask, margin_index]
    collision = collision & valid
    selected = arrays["selected_index"][mask, margin_index]
    rows = np.arange(len(selected))
    raw_collision = collision[:, 0]
    safe = valid & ~collision
    has_safe = safe.any(axis=1)
    target = valid & (arrays["candidate_family"][mask] == "target")
    has_safe_target = (safe & target).any(axis=1)
    rescued = raw_collision & has_safe
    target_rescued = raw_collision & has_safe_target
    incremental = rescued & ~has_safe_target
    unresolved = raw_collision & ~has_safe
    intervention = selected != 0
    terminal = arrays["candidate_velocity"][mask, :, -1]
    progress = arrays["candidate_distance"][mask, :, -1]
    selected_terminal = terminal[rows, selected]
    selected_progress = progress[rows, selected]
    raw_progress = progress[:, 0]
    selected_names = arrays["candidate_name"][mask][rows, selected]
    selected_families = arrays["candidate_family"][mask][rows, selected]
    profile_collision_count = int(collision.sum())
    profile_count = int(valid.sum())
    retained = selected_progress[rescued] / np.maximum(raw_progress[rescued], 1e-6)
    return {
        "n": len(selected),
        "route_ade": float(arrays["route_ade"][mask].mean()),
        "raw_collision_rate": float(raw_collision.mean()),
        "target_only_rescue_count": int(target_rescued.sum()),
        "target_only_rescue_rate_given_raw_collision": _safe_div(
            int(target_rescued.sum()), int(raw_collision.sum())
        ),
        "expanded_rescue_count": int(rescued.sum()),
        "expanded_rescue_rate_given_raw_collision": _safe_div(
            int(rescued.sum()), int(raw_collision.sum())
        ),
        "incremental_profile_rescue_count": int(incremental.sum()),
        "incremental_profile_rescue_rate_given_raw_collision": _safe_div(
            int(incremental.sum()), int(raw_collision.sum())
        ),
        "all_candidates_unsafe_count": int(unresolved.sum()),
        "all_candidates_unsafe_rate": float(unresolved.mean()),
        "oracle_collision_rate": float(unresolved.mean()),
        "oracle_intervention_rate": float(intervention.mean()),
        "nonstop_rescue_rate_given_raw_collision": _safe_div(
            int((rescued & (selected_terminal > 1e-2)).sum()),
            int(raw_collision.sum()),
        ),
        "stopping_rescue_rate_given_raw_collision": _safe_div(
            int((rescued & (selected_terminal <= 1e-2)).sum()),
            int(raw_collision.sum()),
        ),
        "unresolved_zero_target_count": int(
            (unresolved & (arrays["raw_target_speed"][mask] <= 1e-2)).sum()
        ),
        "intervention_rate_on_expert_go": _safe_div(
            int((intervention & ~arrays["expert_brake"][mask]).sum()),
            int((~arrays["expert_brake"][mask]).sum()),
        ),
        "candidate_profile_collision_rate": _safe_div(
            profile_collision_count, profile_count
        ),
        "candidate_count": {
            "mean": float(valid.sum(axis=1).mean()),
            "min": int(valid.sum(axis=1).min()),
            "max": int(valid.sum(axis=1).max()),
        },
        "speed_mps": {
            "current_mean": float(arrays["current_speed"][mask].mean()),
            "raw_target_mean": float(arrays["raw_target_speed"][mask].mean()),
            "expert_target_mean": float(arrays["expert_target_speed"][mask].mean()),
            "raw_terminal_mean": float(terminal[:, 0].mean()),
            "selected_terminal_mean": float(selected_terminal.mean()),
        },
        "progress_m": {
            "raw_mean": float(raw_progress.mean()),
            "selected_mean": float(selected_progress.mean()),
            "reduction_on_rescued": _quantiles(
                raw_progress[rescued] - selected_progress[rescued]
            ),
            "retained_fraction_on_rescued": _quantiles(retained),
        },
        "selected_profile_family": dict(Counter(selected_families.tolist())),
        "selected_profile_on_rescued": dict(Counter(selected_names[rescued].tolist())),
        "raw_collision_ttc_s": _quantiles(
            arrays["candidate_ttc"][mask, margin_index, 0][raw_collision]
        ),
        "raw_route_overflow_rate": float(arrays["candidate_overflow"][mask, 0].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--ckpt-name", default="model_0019.pth")
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--future-cache-dir", required=True)
    parser.add_argument(
        "--anchor-source", choices=("predicted", "cached"), default="predicted"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--speed-fractions", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--stop-times", default="0.5,1,1.5,2")
    parser.add_argument("--safety-margins", default="0,0.1,0.2")
    parser.add_argument(
        "--collision-scope", choices=("dynamic", "all"), default="dynamic"
    )
    parser.add_argument(
        "--route-end-policy", choices=("extrapolate", "clip"), default="extrapolate"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.future_actor_cache import (
        FUTURE_TIMES_S,
        unpack_future_actor_frame,
    )
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.tfv6.future_collision import (
        future_collision_label,
        interpolate_route_by_distance,
        speed_profile_distances,
    )
    from lead.tfv6.tfv6 import TFv6
    from lead.training.config_training import TrainingConfig

    fractions = _parse_float_tuple(args.speed_fractions, minimum=0.0)
    if any(value > 1.0 for value in fractions):
        raise ValueError("--speed-fractions cannot exceed 1")
    stop_times = _parse_float_tuple(args.stop_times, minimum=1e-6)
    margins = _parse_float_tuple(args.safety_margins, minimum=0.0)
    checkpoint = Path(args.ckpt_dir) / args.ckpt_name
    with (Path(args.ckpt_dir) / "config.json").open() as handle:
        config = TrainingConfig(json.load(handle), raise_error_on_missing_key=False)
    if not config.multimodal_planner:
        raise ValueError("velocity-profile oracle requires a multimodal checkpoint")
    config.route_selection_mode = "confidence"
    config.route_speed_safety_gate = False
    config.route_future_safety_gate = False
    config.use_sensor_perburtation = False

    with open(args.heldout_manifest) as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    entry_by_route_frame = {
        (entry["route"], entry["frame"]): entry for entry in entries
    }
    if len(entry_by_route_frame) != len(entries):
        raise ValueError(f"duplicate route/frame entries in {args.heldout_manifest}")

    shards = {}
    cache_key_location = {}
    for path in sorted(Path(args.future_cache_dir).glob("future_actors_*.npz")):
        shard = np.load(path, allow_pickle=False)
        shards[path] = shard
        for index, key in enumerate(shard["keys"]):
            cache_key_location[str(key)] = (path, index)
    cached_route_frames = {
        (entry["route"], entry["frame"])
        for entry in entries
        if entry["key"] in cache_key_location
    }
    if not cached_route_frames:
        raise FileNotFoundError(
            f"no matching future actor cache in {args.future_cache_dir}"
        )

    import timm

    create_model = timm.create_model
    timm.create_model = lambda *a, **k: create_model(*a, **{**k, "pretrained": False})
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        config.use_mixed_precision_training = False
    model = TFv6(device, config).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    model.eval().requires_grad_(False)
    config.detect_boxes = False
    config.use_semantic = False
    config._loaded_config["use_depth"] = False
    config.use_bev_semantic = False

    base = CARLAData(root=config.carla_data, config=config, random=False)
    dataset = VLMIntentDataset(
        base,
        vlm_cache_dir=config.vlm_cache_dir,
        manifest_path=args.heldout_manifest,
        anchor_cache_dir=(
            config.anchor_cache_dir if args.anchor_source == "cached" else None
        ),
    )
    dataset.valid_indices = [
        index
        for index in dataset.valid_indices
        if (
            str(base.images[index], encoding="utf-8").split("/")[-3],
            str(base.images[index], encoding="utf-8").split("/")[-1].split(".")[0],
        )
        in cached_route_frames
    ]
    if args.limit is not None:
        dataset.valid_indices = dataset.valid_indices[: args.limit]
    if not dataset.valid_indices:
        raise ValueError("velocity-profile oracle evaluation set is empty")
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": _collate,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1
    loader = DataLoader(**loader_kwargs)

    n = len(dataset.valid_indices)
    steps = len(FUTURE_TIMES_S)
    max_candidates = (
        1 + len(fractions) + len(config.target_speed_classes) + 4 + len(stop_times)
    )
    route_points = int(config.num_route_points_prediction)
    num_margins = len(margins)
    arrays = {
        "key": np.full(n, "", dtype="U160"),
        "multi": np.zeros(n, dtype=bool),
        "route": np.zeros((n, route_points, 2), dtype=np.float32),
        "route_ade": np.zeros(n, dtype=np.float32),
        "route_length": np.zeros(n, dtype=np.float32),
        "current_speed": np.zeros(n, dtype=np.float32),
        "raw_target_speed": np.zeros(n, dtype=np.float32),
        "expert_target_speed": np.zeros(n, dtype=np.float32),
        "expert_brake": np.zeros(n, dtype=bool),
        "candidate_valid": np.zeros((n, max_candidates), dtype=bool),
        "candidate_name": np.full((n, max_candidates), "", dtype="U40"),
        "candidate_family": np.full((n, max_candidates), "", dtype="U12"),
        "candidate_target_speed": np.full(
            (n, max_candidates), np.nan, dtype=np.float32
        ),
        "candidate_velocity": np.zeros((n, max_candidates, steps), dtype=np.float32),
        "candidate_distance": np.zeros((n, max_candidates, steps), dtype=np.float32),
        "candidate_overflow": np.zeros((n, max_candidates), dtype=bool),
        "candidate_collision": np.zeros((n, num_margins, max_candidates), dtype=bool),
        "candidate_ttc": np.full(
            (n, num_margins, max_candidates), np.inf, dtype=np.float32
        ),
        "selected_index": np.zeros((n, num_margins), dtype=np.int16),
    }
    include_class_ids = (1, 2) if args.collision_scope == "dynamic" else None
    extrapolate = args.route_end_policy == "extrapolate"
    cursor = 0
    for data in tqdm(loader, desc="B3a-0b velocity-profile oracle"):
        if args.anchor_source == "predicted":
            data.pop("anchor", None)
        with (
            torch.inference_mode(),
            torch.amp.autocast(
                device_type=device.type,
                dtype=config.torch_float_type,
                enabled=config.use_mixed_precision_training and device.type == "cuda",
            ),
        ):
            prediction = model(data)
        routes = prediction.pred_route.float().cpu().numpy()
        anchors = prediction.pred_route_anchor
        if anchors is None:
            multi = np.zeros(len(routes), dtype=bool)
        else:
            multi = (anchors[..., 3].cpu().numpy() > 0.5).sum(axis=1) > 1
        labels = data["route"].float().cpu().numpy()
        route_ade = np.linalg.norm(routes - labels, axis=-1).mean(axis=-1)
        current_speeds = data["speed"].float().reshape(-1).cpu().numpy()
        raw_speeds = prediction.pred_target_speed_scalar.float().reshape(-1)
        brake_probability = prediction.pred_target_speed_distribution.float().softmax(
            dim=1
        )[:, 0]
        raw_speeds = (
            torch.where(
                brake_probability > 0.9, torch.zeros_like(raw_speeds), raw_speeds
            )
            .cpu()
            .numpy()
        )
        expert_brakes = data["brake"].bool().reshape(-1).cpu().numpy()
        expert_speeds = data["target_speed"].float().reshape(-1).cpu().numpy()
        expert_speeds = np.where(expert_brakes, 0.0, expert_speeds)

        for batch_index, (route_name, frame_name) in enumerate(
            zip(data["route_number"], data["frame_number"], strict=True)
        ):
            row = cursor + batch_index
            entry = entry_by_route_frame[(route_name, frame_name)]
            shard_path, local_index = cache_key_location[entry["key"]]
            actors = unpack_future_actor_frame(shards[shard_path], local_index)
            route = routes[batch_index]
            profiles = build_velocity_profiles(
                float(current_speeds[batch_index]),
                float(raw_speeds[batch_index]),
                [float(value) for value in config.target_speed_classes],
                fractions,
                stop_times,
                FUTURE_TIMES_S,
                speed_profile_distances,
            )
            if len(profiles) > max_candidates:
                raise RuntimeError(
                    f"candidate buffer too small: {len(profiles)} > {max_candidates}"
                )
            count = len(profiles)
            arrays["key"][row] = entry["key"]
            arrays["multi"][row] = multi[batch_index]
            arrays["route"][row] = route
            arrays["route_ade"][row] = route_ade[batch_index]
            arrays["route_length"][row] = _route_arc_length(route)
            arrays["current_speed"][row] = current_speeds[batch_index]
            arrays["raw_target_speed"][row] = raw_speeds[batch_index]
            arrays["expert_target_speed"][row] = expert_speeds[batch_index]
            arrays["expert_brake"][row] = expert_brakes[batch_index]
            arrays["candidate_valid"][row, :count] = True
            for candidate_index, profile in enumerate(profiles):
                arrays["candidate_name"][row, candidate_index] = profile.name
                arrays["candidate_family"][row, candidate_index] = profile.family
                arrays["candidate_target_speed"][row, candidate_index] = (
                    profile.target_speed
                )
                arrays["candidate_velocity"][row, candidate_index] = profile.velocities
                arrays["candidate_distance"][row, candidate_index] = profile.distances
                arrays["candidate_overflow"][row, candidate_index] = (
                    profile.distances[-1] > arrays["route_length"][row] + 1e-4
                )
                positions, yaws = interpolate_route_by_distance(
                    route, profile.distances, extrapolate=extrapolate
                )
                for margin_index, margin in enumerate(margins):
                    result = future_collision_label(
                        positions,
                        yaws,
                        actors,
                        safety_margin_m=margin,
                        include_class_ids=include_class_ids,
                    )
                    arrays["candidate_collision"][
                        row, margin_index, candidate_index
                    ] = result.collision
                    arrays["candidate_ttc"][row, margin_index, candidate_index] = (
                        result.ttc_s
                    )
            progress = arrays["candidate_distance"][row, :count, -1]
            for margin_index in range(num_margins):
                arrays["selected_index"][row, margin_index] = select_max_progress_safe(
                    arrays["candidate_collision"][row, margin_index, :count], progress
                )
        cursor += len(routes)
    if cursor != n:
        raise RuntimeError(f"evaluated {cursor} frames, expected {n}")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_output = output.with_suffix(".frames.npz")
    np.savez_compressed(
        frame_output,
        **arrays,
        margins_m=np.asarray(margins, dtype=np.float32),
        future_times_s=FUTURE_TIMES_S,
    )
    margin_results = {}
    for margin_index, margin in enumerate(margins):
        scopes = {"all": summarize_margin(arrays, margin_index, np.ones(n, dtype=bool))}
        if arrays["multi"].any():
            scopes["multi"] = summarize_margin(arrays, margin_index, arrays["multi"])
        margin_results[f"{margin:g}"] = scopes
    payload = {
        "checkpoint": str(checkpoint),
        "heldout_manifest": args.heldout_manifest,
        "future_cache_dir": args.future_cache_dir,
        "frame_records": str(frame_output),
        "n_frames": n,
        "route_selection": "highest-confidence valid arm; spatial route never changes",
        "oracle_privilege": "GT future actors choose maximum-progress safe profile",
        "collision_scope": args.collision_scope,
        "route_end_policy": args.route_end_policy,
        "future_times_s": FUTURE_TIMES_S.tolist(),
        "safety_margins_m": list(margins),
        "profile_design": {
            "target_speed_fractions": list(fractions),
            "target_speed_classes_mps": [
                float(value) for value in config.target_speed_classes
            ],
            "constant_acceleration_mps2": [-4.95, -3.0, -1.5, 0.0],
            "stop_times_s": list(stop_times),
            "maximum_acceleration_mps2": 1.89,
            "maximum_deceleration_mps2": 4.95,
            "selection": "maximum 2-second progress among safe candidates; raw retained if safe",
        },
        "margins": margin_results,
    }
    with output.open("w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nB3a-0b velocity-profile oracle ({n} held-out frames; route fixed)")
    for margin in margins:
        result = margin_results[f"{margin:g}"]["all"]
        print(
            f"margin={margin:.2f} raw-coll={result['raw_collision_rate']:.1%} "
            f"target-rescue={result['target_only_rescue_rate_given_raw_collision']:.1%} "
            f"expanded-rescue={result['expanded_rescue_rate_given_raw_collision']:.1%} "
            f"incremental={result['incremental_profile_rescue_count']} "
            f"all-unsafe={result['all_candidates_unsafe_count']} "
            f"oracle-coll={result['oracle_collision_rate']:.1%}"
        )
    print(f"wrote {output}")
    print(f"wrote {frame_output}")


if __name__ == "__main__":
    main()
