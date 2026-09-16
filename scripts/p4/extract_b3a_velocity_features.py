"""Extract frozen winner-route features and counterfactual velocity labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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
}


def _collate(batch: list[dict]) -> dict:
    slim = [
        {key: value for key, value in sample.items() if key in _BATCH_KEYS}
        for sample in batch
    ]
    return torch.utils.data._utils.collate.default_collate(slim)


def _route_arc_length(route: np.ndarray) -> float:
    route = np.asarray(route, dtype=np.float32).reshape(-1, 2)
    if len(route) == 0:
        return 0.0
    if np.linalg.norm(route[0]) > 1e-4:
        route = np.concatenate((np.zeros((1, 2), dtype=np.float32), route), axis=0)
    return float(np.linalg.norm(np.diff(route, axis=0), axis=-1).sum())


def _load_profile_cache(path: Path) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as cache:
        arrays = {name: cache[name] for name in ("keys", "profiles", "current_speed")}
    locations = {str(key): index for index, key in enumerate(arrays["keys"])}
    if len(locations) != len(arrays["keys"]):
        raise ValueError(f"duplicate expert profile keys in {path}")
    return locations, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--ckpt-name", default="model_0019.pth")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--future-cache-dir", required=True)
    parser.add_argument("--expert-profiles", required=True)
    parser.add_argument("--velocity-vocab", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--safety-margin", type=float, default=0.2)
    parser.add_argument("--max-accel", type=float, default=1.89)
    parser.add_argument("--max-decel", type=float, default=4.95)
    parser.add_argument("--overwrite", action="store_true")
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
    from scripts.p4.build_b3a_relative_velocity_vocab import realize_relative_profiles
    from scripts.p4.eval_b3a_velocity_vocab_oracle import (
        interval_speeds_from_distances,
        profile_distances,
        start_reachability_mask,
    )

    checkpoint = Path(args.ckpt_dir) / args.ckpt_name
    with (Path(args.ckpt_dir) / "config.json").open() as handle:
        config = TrainingConfig(json.load(handle), raise_error_on_missing_key=False)
    if not config.multimodal_planner:
        raise ValueError(
            "velocity extraction requires a multimodal corridor checkpoint"
        )
    config.route_selection_mode = "confidence"
    config.route_speed_safety_gate = False
    config.route_future_safety_gate = False
    config.use_sensor_perburtation = False

    with open(args.manifest) as handle:
        manifest = [json.loads(line) for line in handle if line.strip()]
    entry_by_route_frame = {
        (entry["route"], entry["frame"]): entry for entry in manifest
    }
    if len(entry_by_route_frame) != len(manifest):
        raise ValueError(f"duplicate route/frame entries in {args.manifest}")

    actor_shards = {}
    actor_location = {}
    for path in sorted(Path(args.future_cache_dir).glob("future_actors_*.npz")):
        shard = np.load(path, allow_pickle=False)
        actor_shards[path] = shard
        for index, key in enumerate(shard["keys"]):
            actor_location[str(key)] = (path, index)
    missing_actors = [
        entry["key"] for entry in manifest if entry["key"] not in actor_location
    ]
    if missing_actors:
        raise FileNotFoundError(
            f"future actor cache is incomplete: {len(missing_actors)} missing; "
            f"first={missing_actors[0]}"
        )

    profile_location, expert_cache = _load_profile_cache(Path(args.expert_profiles))
    missing_profiles = [
        entry["key"] for entry in manifest if entry["key"] not in profile_location
    ]
    if missing_profiles:
        raise FileNotFoundError(
            f"expert profile cache is incomplete: {len(missing_profiles)} missing; "
            f"first={missing_profiles[0]}"
        )
    vocabulary = np.load(args.velocity_vocab).astype(np.float32)
    if vocabulary.ndim != 2 or vocabulary.shape[1] != len(FUTURE_TIMES_S):
        raise ValueError(
            f"velocity vocabulary has shape {vocabulary.shape}; expected [K,{len(FUTURE_TIMES_S)}]"
        )
    interval_s = float(FUTURE_TIMES_S[0])

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
        manifest_path=args.manifest,
        anchor_cache_dir=None,
    )
    start = max(0, args.start)
    end = (
        len(dataset.valid_indices)
        if args.end is None
        else min(args.end, len(dataset.valid_indices))
    )
    if start >= end:
        raise ValueError(f"empty extraction interval [{start}, {end})")
    dataset.valid_indices = dataset.valid_indices[start:end]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"velocity_features_{start:06d}_{end:06d}.npz"
    if output.exists() and not args.overwrite:
        print(f"exists, skipping {output}")
        return
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": _collate,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1
    loader = DataLoader(**loader_kwargs)

    names = (
        "route_features",
        "current_speed",
        "raw_target_speed",
        "candidate_velocity",
        "candidate_valid",
        "collision",
        "ttc",
        "imitation_error",
        "imitation_target",
        "raw_route_ade",
        "selected_arm",
        "multi",
        "keys",
    )
    saved: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for data in tqdm(loader, desc=f"B3a velocity features {start}:{end}"):
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
        if (
            prediction.pred_route_features is None
            or prediction.pred_route_selected_idx is None
        ):
            raise RuntimeError("corridor model did not expose selected route features")
        routes = prediction.pred_route.float().cpu().numpy()
        all_features = prediction.pred_route_features.float().cpu().numpy()
        selected_arm = prediction.pred_route_selected_idx.long().cpu().numpy()
        rows = np.arange(len(routes))
        route_features = all_features[rows, selected_arm]
        anchors = prediction.pred_route_anchor
        multi = (
            (anchors[..., 3].cpu().numpy() > 0.5).sum(axis=1) > 1
            if anchors is not None
            else np.zeros(len(routes), dtype=bool)
        )
        current_speed = np.maximum(data["speed"].float().reshape(-1).cpu().numpy(), 0.0)
        raw_target_tensor = prediction.pred_target_speed_scalar.float().reshape(-1)
        brake_probability = prediction.pred_target_speed_distribution.float().softmax(
            dim=1
        )[:, 0]
        raw_target_speed = (
            torch.where(
                brake_probability > 0.9,
                torch.zeros_like(raw_target_tensor),
                raw_target_tensor,
            )
            .cpu()
            .numpy()
        )

        batch_keys = []
        expert_velocity = []
        for route_name, frame_name in zip(
            data["route_number"], data["frame_number"], strict=True
        ):
            entry = entry_by_route_frame[(route_name, frame_name)]
            key = entry["key"]
            batch_keys.append(key)
            expert_index = profile_location[key]
            expert_velocity.append(expert_cache["profiles"][expert_index])
            cached_speed = max(0.0, float(expert_cache["current_speed"][expert_index]))
            if not np.isclose(
                cached_speed, current_speed[len(batch_keys) - 1], atol=1e-3
            ):
                raise ValueError(f"current-speed mismatch for {key}")
        expert_velocity = np.asarray(expert_velocity, dtype=np.float32)

        relative_velocity = realize_relative_profiles(vocabulary, current_speed)
        relative_distance = profile_distances(relative_velocity, interval_s)
        relative_valid = start_reachability_mask(
            relative_velocity,
            current_speed,
            interval_s,
            max_accel_mps2=args.max_accel,
            max_decel_mps2=args.max_decel,
        )
        raw_distance = np.stack(
            [
                speed_profile_distances(float(v0), float(target))
                for v0, target in zip(current_speed, raw_target_speed, strict=True)
            ]
        )
        raw_velocity = interval_speeds_from_distances(raw_distance, interval_s)
        candidate_velocity = np.concatenate(
            (raw_velocity[:, None], relative_velocity), axis=1
        )
        candidate_distance = np.concatenate(
            (raw_distance[:, None], relative_distance), axis=1
        )
        candidate_valid = np.concatenate(
            (np.ones((len(routes), 1), dtype=bool), relative_valid), axis=1
        )
        imitation_error = np.abs(expert_velocity[:, None] - candidate_velocity).mean(
            axis=2
        )
        imitation_target = np.where(candidate_valid, imitation_error, np.inf).argmin(
            axis=1
        )
        collision = np.zeros_like(candidate_valid)
        ttc = np.full(candidate_valid.shape, np.inf, dtype=np.float32)

        for batch_index, (key, route) in enumerate(
            zip(batch_keys, routes, strict=True)
        ):
            shard_path, local_index = actor_location[key]
            actors = unpack_future_actor_frame(actor_shards[shard_path], local_index)
            for candidate_index in np.flatnonzero(candidate_valid[batch_index]):
                positions, yaws = interpolate_route_by_distance(
                    route,
                    candidate_distance[batch_index, candidate_index],
                    extrapolate=True,
                )
                label = future_collision_label(
                    positions,
                    yaws,
                    actors,
                    safety_margin_m=args.safety_margin,
                    include_class_ids=(1, 2),
                )
                collision[batch_index, candidate_index] = label.collision
                ttc[batch_index, candidate_index] = label.ttc_s

        route_label = data["route"].float().cpu().numpy()
        raw_route_ade = np.linalg.norm(routes - route_label, axis=-1).mean(axis=-1)
        values = {
            "route_features": route_features.astype(np.float16),
            "current_speed": current_speed.astype(np.float16),
            "raw_target_speed": raw_target_speed.astype(np.float16),
            "candidate_velocity": candidate_velocity.astype(np.float16),
            "candidate_valid": candidate_valid,
            "collision": collision,
            "ttc": ttc.astype(np.float16),
            "imitation_error": imitation_error.astype(np.float16),
            "imitation_target": imitation_target.astype(np.int16),
            "raw_route_ade": raw_route_ade.astype(np.float16),
            "selected_arm": selected_arm.astype(np.int8),
            "multi": multi,
            "keys": np.asarray(batch_keys),
        }
        for name in names:
            saved[name].append(values[name])

    packed = {name: np.concatenate(parts, axis=0) for name, parts in saved.items()}
    vocab_hash = hashlib.sha256(Path(args.velocity_vocab).read_bytes()).hexdigest()
    packed.update(
        source_checkpoint=np.asarray(str(checkpoint.resolve())),
        source_manifest=np.asarray(str(Path(args.manifest).resolve())),
        source_vocabulary=np.asarray(str(Path(args.velocity_vocab).resolve())),
        vocabulary_sha256=np.asarray(vocab_hash),
        safety_margin_m=np.asarray(args.safety_margin, dtype=np.float32),
        candidate_zero=np.asarray("raw target-speed profile"),
    )
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **packed)
    os.replace(temporary, output)
    valid_labels = packed["collision"][packed["candidate_valid"]]
    print(
        f"wrote {output}: frames={len(packed['keys'])} "
        f"candidates={packed['candidate_velocity'].shape[1]} "
        f"valid/frame={packed['candidate_valid'].sum(1).mean():.1f} "
        f"collision={valid_labels.mean():.3%}"
    )


if __name__ == "__main__":
    main()
