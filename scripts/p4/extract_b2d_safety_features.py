"""Extract frozen corridor arm features and privileged dynamic-safety labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


_BATCH_KEYS = {
    "rgb", "rasterized_lidar", "vlm_hidden", "radar", "speed", "command",
    "target_point", "target_point_previous", "target_point_next", "route",
    "route_number", "frame_number", "anchor",
}


def _collate(batch: list[dict]) -> dict:
    slim = [{key: value for key, value in sample.items() if key in _BATCH_KEYS} for sample in batch]
    return torch.utils.data._utils.collate.default_collate(slim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--ckpt-name", default="model_0019.pth")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--future-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--safety-margin", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.future_actor_cache import unpack_future_actor_frame
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.tfv6.future_collision import route_future_collision_label
    from lead.tfv6.tfv6 import TFv6
    from lead.training.config_training import TrainingConfig

    checkpoint = Path(args.ckpt_dir) / args.ckpt_name
    with (Path(args.ckpt_dir) / "config.json").open() as handle:
        config = TrainingConfig(json.load(handle), raise_error_on_missing_key=False)
    if not config.multimodal_planner:
        raise ValueError("B2d extraction requires a multimodal corridor checkpoint")
    config.route_selection_mode = "confidence"
    config.route_speed_safety_gate = False
    config.use_sensor_perburtation = False

    with open(args.manifest) as handle:
        manifest = [json.loads(line) for line in handle if line.strip()]
    entry_by_route_frame = {(entry["route"], entry["frame"]): entry for entry in manifest}
    if len(entry_by_route_frame) != len(manifest):
        raise ValueError(f"duplicate route/frame entries in {args.manifest}")

    actor_shards = {}
    cache_key_location = {}
    for path in sorted(Path(args.future_cache_dir).glob("future_actors_*.npz")):
        shard = np.load(path, allow_pickle=False)
        actor_shards[path] = shard
        for index, key in enumerate(shard["keys"]):
            cache_key_location[str(key)] = (path, index)
    missing = [entry["key"] for entry in manifest if entry["key"] not in cache_key_location]
    if missing:
        raise FileNotFoundError(
            f"future actor cache is incomplete: {len(missing)} / {len(manifest)} keys missing; "
            f"first={missing[0]}"
        )

    import timm
    create_model = timm.create_model
    timm.create_model = lambda *a, **k: create_model(*a, **{**k, "pretrained": False})
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        config.use_mixed_precision_training = False
    model = TFv6(device, config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model.eval().requires_grad_(False)

    # They run after planning and cannot affect the cached arm representation.
    config.detect_boxes = False
    config.use_semantic = False
    config._loaded_config["use_depth"] = False
    config.use_bev_semantic = False

    base = CARLAData(root=config.carla_data, config=config, random=False)
    dataset = VLMIntentDataset(
        base,
        vlm_cache_dir=config.vlm_cache_dir,
        manifest_path=args.manifest,
        anchor_cache_dir=None,  # predicted VLM anchors, matching closed-loop inference
    )
    start = max(0, args.start)
    end = len(dataset.valid_indices) if args.end is None else min(args.end, len(dataset.valid_indices))
    if start >= end:
        raise ValueError(f"empty extraction interval [{start}, {end})")
    dataset.valid_indices = dataset.valid_indices[start:end]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"safety_features_{start:06d}_{end:06d}.npz"
    if output.exists() and not args.overwrite:
        print(f"exists, skipping {output}")
        return

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1
    loader = DataLoader(**loader_kwargs)
    names = (
        "route_features", "confidence", "valid", "collision", "ttc", "ade",
        "current_speed", "target_speed", "keys",
    )
    saved: dict[str, list[np.ndarray]] = {name: [] for name in names}

    for data in tqdm(loader, desc=f"B2d features {start}:{end}"):
        # VLM intent is frozen/deterministic; omitting the dataset anchor forces the exact
        # predicted-anchor path used by closed-loop inference and the preceding audit.
        data.pop("anchor", None)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type,
            dtype=config.torch_float_type,
            enabled=config.use_mixed_precision_training and device.type == "cuda",
        ):
            prediction = model(data)
        if prediction.pred_route_features is None:
            raise RuntimeError("corridor model did not expose pred_route_features")
        routes = prediction.pred_route_multimodal.float().cpu().numpy()
        features = prediction.pred_route_features.float().cpu().numpy()
        confidence = prediction.pred_route_conf.float().cpu().numpy()
        anchors = prediction.pred_route_anchor
        valid = anchors[..., 3].cpu().numpy() > 0.5
        all_padding = ~valid.any(axis=1)
        valid[all_padding, 0] = True
        expert_route = data["route"].float().cpu().numpy()
        ade = np.linalg.norm(routes - expert_route[:, None], axis=-1).mean(axis=-1)

        raw_target_speed = prediction.pred_target_speed_scalar.float().reshape(-1)
        brake_probability = prediction.pred_target_speed_distribution.float().softmax(dim=1)[:, 0]
        raw_target_speed = torch.where(
            brake_probability > 0.9, torch.zeros_like(raw_target_speed), raw_target_speed,
        )
        target_speed = raw_target_speed.cpu().numpy()
        current_speed = data["speed"].float().reshape(-1).cpu().numpy()
        collision = np.zeros_like(valid, dtype=bool)
        ttc = np.full_like(confidence, np.inf, dtype=np.float32)
        batch_keys = []
        for batch_index, (route_name, frame_name) in enumerate(
            zip(data["route_number"], data["frame_number"], strict=True),
        ):
            entry = entry_by_route_frame[(route_name, frame_name)]
            batch_keys.append(entry["key"])
            shard_path, local_index = cache_key_location[entry["key"]]
            actors = unpack_future_actor_frame(actor_shards[shard_path], local_index)
            for arm_index in np.flatnonzero(valid[batch_index]):
                label = route_future_collision_label(
                    routes[batch_index, arm_index],
                    float(current_speed[batch_index]),
                    float(target_speed[batch_index]),
                    actors,
                    safety_margin_m=args.safety_margin,
                    include_class_ids=(1, 2),
                )
                collision[batch_index, arm_index] = label.collision
                ttc[batch_index, arm_index] = label.ttc_s

        values = {
            "route_features": features.astype(np.float16),
            "confidence": confidence.astype(np.float16),
            "valid": valid,
            "collision": collision,
            "ttc": ttc.astype(np.float16),
            "ade": ade.astype(np.float16),
            "current_speed": current_speed.astype(np.float16),
            "target_speed": target_speed.astype(np.float16),
            "keys": np.asarray(batch_keys),
        }
        for name in names:
            saved[name].append(values[name])

    packed = {name: np.concatenate(parts, axis=0) for name, parts in saved.items()}
    packed.update(
        source_checkpoint=np.asarray(str(checkpoint)),
        source_manifest=np.asarray(str(args.manifest)),
        safety_margin_m=np.asarray(args.safety_margin, dtype=np.float32),
    )
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **packed)
    os.replace(temporary, output)
    print(
        f"wrote {output}: frames={len(packed['keys'])} "
        f"valid_arms={int(packed['valid'].sum())} "
        f"collision_rate={packed['collision'][packed['valid']].mean():.3%}"
    )


if __name__ == "__main__":
    main()
