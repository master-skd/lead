"""Cache frozen LEAD CenterNet actor detections for held-out velocity gating."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def _sample_feature_keys(feature_dir: Path, limit: int) -> list[str]:
    keys = []
    for path in sorted(feature_dir.glob("velocity_features_*.npz")):
        with np.load(path, allow_pickle=False) as shard:
            keys.extend(shard["keys"].tolist())
    if not keys:
        raise FileNotFoundError(f"no held-out velocity features in {feature_dir}")
    if len(set(keys)) != len(keys):
        raise ValueError("held-out feature keys are not unique")
    if limit < 1:
        raise ValueError("limit must be positive")
    if limit >= len(keys):
        return keys
    indices = np.linspace(0, len(keys) - 1, limit, dtype=np.int64)
    return [keys[index] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--ckpt-name", default="model_0019.pth")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--nearest-vlm-manifest", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.shard_size < 1 or args.batch_size < 1:
        parser.error("shard-size and batch-size must be positive")

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from lead.data_loader import carla_dataset_utils
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.tfv6.tfv6 import TFv6
    from lead.training.config_training import TrainingConfig
    from scripts.p4.extract_b3a_velocity_features import _collate

    checkpoint = Path(args.ckpt_dir) / args.ckpt_name
    with (Path(args.ckpt_dir) / "config.json").open() as handle:
        saved_config = json.load(handle)
    model_config = TrainingConfig(saved_config, raise_error_on_missing_key=False)
    model_config.route_selection_mode = "confidence"
    model_config.route_speed_safety_gate = False
    model_config.route_future_safety_gate = False
    model_config.use_sensor_perburtation = False
    if not model_config.detect_boxes:
        raise ValueError("the frozen checkpoint has no CenterNet detection head")
    if model_config.training_used_lidar_steps <= 1:
        raise ValueError("the frozen detection head does not predict actor speed")

    import timm

    create_model = timm.create_model
    timm.create_model = lambda *a, **k: create_model(*a, **{**k, "pretrained": False})
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CenterNet extraction requires a CUDA GPU")
    model = TFv6(device, model_config).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    model.eval().requires_grad_(False)
    model_config.use_semantic = False
    model_config._loaded_config["use_depth"] = False
    model_config.use_bev_semantic = False

    dataset_config = TrainingConfig(saved_config, raise_error_on_missing_key=False)
    dataset_config.defer_vlm_inputs_to_wrapper = True
    dataset_config.use_sensor_perburtation = False
    dataset_config.detect_boxes = False
    dataset_config.use_semantic = False
    dataset_config._loaded_config["use_depth"] = False
    dataset_config.use_bev_semantic = False
    base = CARLAData(
        root=dataset_config.carla_data, config=dataset_config, random=False
    )
    dataset = VLMIntentDataset(
        base,
        vlm_cache_dir=dataset_config.vlm_cache_dir,
        manifest_path=args.manifest,
        anchor_cache_dir=None,
        nearest_cache_manifest_path=args.nearest_vlm_manifest,
    )
    selected_keys = _sample_feature_keys(Path(args.feature_cache_dir), args.limit)
    wanted = set(selected_keys)
    index_by_key = {}
    for index in dataset.valid_indices:
        image_path = str(base.images[index], encoding="utf-8")
        parts = image_path.split("/")
        key = f"{parts[-4]}__{parts[-3]}__{parts[-1].split('.')[0]}"
        if key in wanted:
            index_by_key[key] = index
    missing = wanted - index_by_key.keys()
    if missing:
        raise ValueError(
            f"{len(missing)} sampled feature frames missing; first={min(missing)}"
        )
    dataset.valid_indices = [index_by_key[key] for key in selected_keys]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(selected_keys), args.shard_size):
        end = min(start + args.shard_size, len(selected_keys))
        output = output_dir / f"predicted_boxes_{start:06d}_{end:06d}.npz"
        if output.exists() and not args.overwrite:
            with np.load(output, allow_pickle=False) as cached:
                if cached["keys"].tolist() != selected_keys[start:end]:
                    raise ValueError(
                        f"cached keys do not match current sample: {output}"
                    )
                if str(cached["source_checkpoint"]) != str(checkpoint.resolve()):
                    raise ValueError(f"cached checkpoint does not match: {output}")
            print(f"exists, skipping {output}")
            continue
        loader = DataLoader(
            Subset(dataset, range(start, end)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=_collate,
            pin_memory=True,
        )
        boxes = []
        seen_keys = []
        for data in tqdm(loader, desc=f"CenterNet boxes {start}:{end}"):
            data.pop("anchor", None)
            with (
                torch.inference_mode(),
                torch.amp.autocast(
                    device_type="cuda",
                    dtype=model_config.torch_float_type,
                    enabled=model_config.use_mixed_precision_training,
                ),
            ):
                prediction = model(data)
            if prediction.pred_bounding_box is None:
                raise RuntimeError("CenterNet returned no boxes")
            image_boxes = prediction.pred_bounding_box.pred_bounding_box_image_system
            for raw, route, frame in zip(
                image_boxes,
                data["route_number"],
                data["frame_number"],
                strict=True,
            ):
                vehicle_boxes = carla_dataset_utils.bb_image_to_vehicle_system(
                    raw,
                    model_config.pixels_per_meter,
                    model_config.min_x_meter,
                    model_config.min_y_meter,
                )
                boxes.append(vehicle_boxes.astype(np.float32))
                seen_keys.append((route, frame))
        expected = selected_keys[start:end]
        actual_pairs = [tuple(key.rsplit("__", 2)[1:]) for key in expected]
        if seen_keys != actual_pairs:
            raise ValueError(
                f"model output order does not match feature keys: {output}"
            )
        packed = {
            "keys": np.asarray(expected),
            "boxes": np.stack(boxes),
            "source_checkpoint": np.asarray(str(checkpoint.resolve())),
            "source_features": np.asarray(str(Path(args.feature_cache_dir).resolve())),
            "sample_limit": np.asarray(args.limit, dtype=np.int32),
        }
        temporary = output.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **packed)
        os.replace(temporary, output)
        print(
            f"wrote {output}: {len(expected)} frames, {packed['boxes'].shape[1]} top-K boxes"
        )


if __name__ == "__main__":
    main()
