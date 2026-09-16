"""No-training velocity-lattice oracle on the fixed confidence-winner route.

This is an upper-bound audit, not a deployable selector: GT future actors decide
which physically reachable target-speed profile is safe.  The spatial route and
navigation branch are never changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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


def _collate(batch: list[dict]) -> dict:
    slim = [
        {key: value for key, value in sample.items() if key in _BATCH_KEYS}
        for sample in batch
    ]
    return torch.utils.data._utils.collate.default_collate(slim)


def _parse_fractions(value: str) -> tuple[float, ...]:
    fractions = tuple(float(item) for item in value.split(",") if item.strip())
    if not fractions or any(item < 0.0 or item > 1.0 for item in fractions):
        raise ValueError("--speed-fractions must contain values in [0,1]")
    return fractions


def build_target_speed_candidates(
    raw_target_speed: float,
    speed_classes: list[float] | tuple[float, ...],
    fractions: tuple[float, ...],
    *,
    include_speed_classes: bool = True,
) -> tuple[np.ndarray, int]:
    """Return sorted slowdown targets and the index of the untouched raw target."""

    raw = max(0.0, float(raw_target_speed))
    values = [raw, *(raw * fraction for fraction in fractions)]
    if include_speed_classes:
        values.extend(
            float(speed) for speed in speed_classes if 0.0 <= float(speed) <= raw + 1e-6
        )
    candidates = np.unique(np.round(np.asarray(values, dtype=np.float32), decimals=6))
    raw_index = int(np.abs(candidates - raw).argmin())
    if not np.isclose(candidates[raw_index], raw, atol=1e-5):
        raise RuntimeError("raw target speed was lost while building candidates")
    return candidates, raw_index


def select_minimal_slowdown(
    candidate_speeds: np.ndarray,
    collision: np.ndarray,
    raw_index: int,
) -> int:
    """Keep a safe raw profile, otherwise choose the fastest safe slowdown."""

    if not bool(collision[raw_index]):
        return int(raw_index)
    safe = np.flatnonzero(~np.asarray(collision, dtype=bool))
    if len(safe) == 0:
        return int(raw_index)
    return int(safe[np.argmax(candidate_speeds[safe])])


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator / max(denominator, 1))


def _route_arc_length(route: np.ndarray) -> float:
    route = np.asarray(route, dtype=np.float32).reshape(-1, 2)
    if len(route) == 0:
        return 0.0
    if np.linalg.norm(route[0]) > 1e-4:
        route = np.concatenate([np.zeros((1, 2), dtype=np.float32), route], axis=0)
    return float(np.linalg.norm(np.diff(route, axis=0), axis=-1).sum())


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


def summarize_velocity_oracle(records: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    values = {key: np.asarray(value)[mask] for key, value in records.items()}
    n = len(values["raw_collision"])
    raw_collision = values["raw_collision"].astype(bool)
    selected_collision = values["selected_collision"].astype(bool)
    rescued = raw_collision & ~selected_collision
    intervention = values["selected_speed"] < values["raw_speed"] - 1e-4
    moving_rescue = rescued & (values["selected_speed"] > 1e-2)
    stop_rescue = rescued & (values["selected_speed"] <= 1e-2)
    expert_brake = values["expert_brake"].astype(bool)
    expert_go = ~expert_brake
    reduction = values["raw_speed"] - values["selected_speed"]
    distribution = Counter(np.round(values["selected_speed"], 1).astype(str).tolist())
    return {
        "n": n,
        "route_ade": float(values["route_ade"].mean()),
        "mean_route_length_m": float(values["route_length"].mean()),
        "candidate_count": {
            "mean": float(values["candidate_count"].mean()),
            "min": int(values["candidate_count"].min()),
            "max": int(values["candidate_count"].max()),
        },
        "candidate_profile_collision_rate": _safe_div(
            int(values["candidate_collision_count"].sum()),
            int(values["candidate_count"].sum()),
        ),
        "raw_collision_rate": float(raw_collision.mean()),
        "expert_speed_profile_collision_rate": float(values["expert_collision"].mean()),
        "all_candidates_unsafe_rate": float(values["all_candidates_unsafe"].mean()),
        "rescue_possible_rate": float(rescued.mean()),
        "rescue_rate_given_raw_collision": _safe_div(
            int(rescued.sum()), int(raw_collision.sum())
        ),
        "oracle_collision_rate": float(selected_collision.mean()),
        "oracle_intervention_rate": float(intervention.mean()),
        "intervention_rate_given_raw_collision": _safe_div(
            int(intervention.sum()),
            int(raw_collision.sum()),
        ),
        "moving_rescue_rate_given_raw_collision": _safe_div(
            int(moving_rescue.sum()),
            int(raw_collision.sum()),
        ),
        "stop_rescue_rate_given_raw_collision": _safe_div(
            int(stop_rescue.sum()),
            int(raw_collision.sum()),
        ),
        "only_stop_is_safe_rate_given_raw_collision": _safe_div(
            int(values["only_stop_safe"].sum()),
            int(raw_collision.sum()),
        ),
        "rescue_rate_on_expert_brake_collision": _safe_div(
            int((rescued & expert_brake).sum()),
            int((raw_collision & expert_brake).sum()),
        ),
        "oracle_intervention_rate_on_expert_go": _safe_div(
            int((intervention & expert_go).sum()),
            int(expert_go.sum()),
        ),
        "speed_mps": {
            "raw_mean": float(values["raw_speed"].mean()),
            "selected_mean": float(values["selected_speed"].mean()),
            "expert_mean": float(values["expert_speed"].mean()),
            "raw_mae_to_expert": float(
                np.abs(values["raw_speed"] - values["expert_speed"]).mean()
            ),
            "selected_mae_to_expert": float(
                np.abs(values["selected_speed"] - values["expert_speed"]).mean()
            ),
            "reduction_on_rescued": _quantiles(reduction[rescued]),
            "selected_target_histogram_0p1mps": dict(
                sorted(distribution.items(), key=lambda item: float(item[0]))
            ),
        },
        "route_horizon": {
            "raw_profile_overflow_rate": float(values["raw_overflow"].mean()),
            "candidate_profile_overflow_rate": _safe_div(
                int(values["candidate_overflow_count"].sum()),
                int(values["candidate_count"].sum()),
            ),
        },
        "raw_collision_ttc_s": _quantiles(values["raw_ttc"]),
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
    parser.add_argument("--safety-margin", type=float, default=0.2)
    parser.add_argument("--speed-fractions", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--exclude-speed-classes", action="store_true")
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
    from lead.data_loader.future_actor_cache import unpack_future_actor_frame
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.tfv6.future_collision import (
        route_future_collision_label,
        speed_profile_distances,
    )
    from lead.tfv6.tfv6 import TFv6
    from lead.training.config_training import TrainingConfig

    checkpoint = Path(args.ckpt_dir) / args.ckpt_name
    with (Path(args.ckpt_dir) / "config.json").open() as handle:
        config = TrainingConfig(json.load(handle), raise_error_on_missing_key=False)
    if not config.multimodal_planner:
        raise ValueError("velocity oracle requires a multimodal corridor checkpoint")
    config.route_selection_mode = "confidence"
    config.route_speed_safety_gate = False
    config.route_future_safety_gate = False
    config.use_sensor_perburtation = False

    with open(args.heldout_manifest) as handle:
        heldout_entries = [json.loads(line) for line in handle if line.strip()]
    entry_by_route_frame = {
        (entry["route"], entry["frame"]): entry for entry in heldout_entries
    }
    if len(entry_by_route_frame) != len(heldout_entries):
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
        for entry in heldout_entries
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

    # These heads run after planning and are irrelevant to route/speed prediction.
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
        raise ValueError("velocity-oracle evaluation set is empty")
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

    fractions = _parse_fractions(args.speed_fractions)
    speed_classes = [float(item) for item in config.target_speed_classes]
    include_class_ids = (1, 2) if args.collision_scope == "dynamic" else None
    extrapolate_route = args.route_end_policy == "extrapolate"
    saved = {
        key: []
        for key in (
            "key",
            "multi",
            "route_ade",
            "route_length",
            "candidate_count",
            "candidate_collision_count",
            "candidate_overflow_count",
            "raw_collision",
            "raw_ttc",
            "raw_overflow",
            "expert_collision",
            "all_candidates_unsafe",
            "only_stop_safe",
            "selected_collision",
            "raw_speed",
            "selected_speed",
            "expert_speed",
            "expert_brake",
        )
    }

    for data in tqdm(loader, desc="B3a-0 velocity oracle"):
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
        raw_speed_tensor = prediction.pred_target_speed_scalar.float().reshape(-1)
        brake_probability = prediction.pred_target_speed_distribution.float().softmax(
            dim=1
        )[:, 0]
        raw_speed_tensor = torch.where(
            brake_probability > 0.9,
            torch.zeros_like(raw_speed_tensor),
            raw_speed_tensor,
        )
        raw_speeds = raw_speed_tensor.cpu().numpy()
        expert_brakes = data["brake"].bool().reshape(-1).cpu().numpy()
        expert_speeds = data["target_speed"].float().reshape(-1).cpu().numpy()
        expert_speeds = np.where(expert_brakes, 0.0, expert_speeds)

        for batch_index, (route_name, frame_name) in enumerate(
            zip(data["route_number"], data["frame_number"], strict=True),
        ):
            entry = entry_by_route_frame[(route_name, frame_name)]
            shard_path, local_index = cache_key_location[entry["key"]]
            actors = unpack_future_actor_frame(shards[shard_path], local_index)
            route = routes[batch_index]
            current_speed = float(current_speeds[batch_index])
            raw_speed = float(raw_speeds[batch_index])
            expert_speed = float(expert_speeds[batch_index])
            candidate_speeds, raw_index = build_target_speed_candidates(
                raw_speed,
                speed_classes,
                fractions,
                include_speed_classes=not args.exclude_speed_classes,
            )
            collision = np.zeros(len(candidate_speeds), dtype=bool)
            ttc = np.full(len(candidate_speeds), np.inf, dtype=np.float32)
            route_length = _route_arc_length(route)
            overflow = np.zeros(len(candidate_speeds), dtype=bool)
            for candidate_index, target_speed in enumerate(candidate_speeds):
                distances = speed_profile_distances(current_speed, float(target_speed))
                overflow[candidate_index] = bool(distances[-1] > route_length + 1e-4)
                result = route_future_collision_label(
                    route,
                    current_speed,
                    float(target_speed),
                    actors,
                    extrapolate_route=extrapolate_route,
                    safety_margin_m=args.safety_margin,
                    include_class_ids=include_class_ids,
                )
                collision[candidate_index] = result.collision
                ttc[candidate_index] = result.ttc_s

            selected_index = select_minimal_slowdown(
                candidate_speeds, collision, raw_index
            )
            zero_index = int(np.abs(candidate_speeds).argmin())
            nonzero_safe = bool((~collision & (candidate_speeds > 1e-2)).any())
            expert_result = route_future_collision_label(
                route,
                current_speed,
                expert_speed,
                actors,
                extrapolate_route=extrapolate_route,
                safety_margin_m=args.safety_margin,
                include_class_ids=include_class_ids,
            )
            frame_values = {
                "key": entry["key"],
                "multi": bool(multi[batch_index]),
                "route_ade": float(route_ade[batch_index]),
                "route_length": route_length,
                "candidate_count": len(candidate_speeds),
                "candidate_collision_count": int(collision.sum()),
                "candidate_overflow_count": int(overflow.sum()),
                "raw_collision": bool(collision[raw_index]),
                "raw_ttc": float(ttc[raw_index]),
                "raw_overflow": bool(overflow[raw_index]),
                "expert_collision": bool(expert_result.collision),
                "all_candidates_unsafe": bool(collision.all()),
                "only_stop_safe": bool(
                    collision[raw_index] and ~collision[zero_index] and not nonzero_safe
                ),
                "selected_collision": bool(collision[selected_index]),
                "raw_speed": raw_speed,
                "selected_speed": float(candidate_speeds[selected_index]),
                "expert_speed": expert_speed,
                "expert_brake": bool(expert_brakes[batch_index]),
            }
            for key, value in frame_values.items():
                saved[key].append(value)

    records = {key: np.asarray(value) for key, value in saved.items()}
    scopes = {
        "all": summarize_velocity_oracle(
            records, np.ones(len(records["multi"]), dtype=bool)
        ),
    }
    if records["multi"].any():
        scopes["multi"] = summarize_velocity_oracle(
            records, records["multi"].astype(bool)
        )
    payload = {
        "checkpoint": str(checkpoint),
        "heldout_manifest": args.heldout_manifest,
        "future_cache_dir": args.future_cache_dir,
        "n_frames": len(records["multi"]),
        "route_selection": "highest-confidence valid arm; spatial route never changes",
        "oracle_privilege": "GT future actors select the fastest safe slowdown only when raw collides",
        "collision_scope": args.collision_scope,
        "dynamic_classes": ["car", "walker"]
        if args.collision_scope == "dynamic"
        else None,
        "safety_margin_m": args.safety_margin,
        "route_end_policy": args.route_end_policy,
        "velocity_candidates": {
            "profile": "constant acceleration toward target, bounded by +1.89/-4.95 m/s^2",
            "horizon_s": 2.0,
            "dt_s": 0.25,
            "raw_speed_fractions": list(fractions),
            "target_speed_classes_mps": speed_classes
            if not args.exclude_speed_classes
            else [],
            "acceleration_candidates": False,
        },
        "scopes": scopes,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_output = output.with_suffix(".frames.npz")
    np.savez_compressed(frame_output, **records)
    payload["frame_records"] = str(frame_output)
    with output.open("w") as handle:
        json.dump(payload, handle, indent=2)

    print(
        f"\nB3a-0 velocity oracle ({len(records['multi'])} held-out frames; route fixed)"
    )
    print(
        f"collision={args.collision_scope}, margin={args.safety_margin:.2f} m, "
        f"route-end={args.route_end_policy}"
    )
    for name, summary in scopes.items():
        speed = summary["speed_mps"]
        print(
            f"[{name}] n={summary['n']} candidates={summary['candidate_count']['mean']:.1f} "
            f"raw-coll={summary['raw_collision_rate']:.1%} "
            f"all-unsafe={summary['all_candidates_unsafe_rate']:.1%} "
            f"rescue={summary['rescue_rate_given_raw_collision']:.1%} "
            f"oracle-coll={summary['oracle_collision_rate']:.1%}"
        )
        print(
            f"  rescue moving/stop={summary['moving_rescue_rate_given_raw_collision']:.1%}/"
            f"{summary['stop_rescue_rate_given_raw_collision']:.1%}; "
            f"target speed {speed['raw_mean']:.2f}->{speed['selected_mean']:.2f} m/s; "
            f"MAE to expert {speed['raw_mae_to_expert']:.3f}->{speed['selected_mae_to_expert']:.3f}; "
            f"raw route overflow={summary['route_horizon']['raw_profile_overflow_rate']:.1%}"
        )
    print(f"wrote {output}")
    print(f"wrote {frame_output}")


if __name__ == "__main__":
    main()
