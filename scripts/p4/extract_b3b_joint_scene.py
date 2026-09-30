"""Extract same-forward scene memory and all B2 Path candidates for joint scoring.

This is the frozen-backbone GPU stage. No counterfactual GT labels are made
here; those are a separate CPU stage over these exact saved candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.p4.eval_b3b_joint_oracle import local_path_variants
from scripts.p4.extract_b3a_velocity_features import _collate

SCHEMA_VERSION = 1
FIELDS = (
    "keys", "scene_tokens", "routes", "route_valid", "route_conf",
    "winner", "current_speed", "raw_target_speed", "expert_route",
    "command", "anchor_fallback",
)


def pack_batch(prediction, data: dict, keys: list[str], offsets: tuple[float, ...],
               *, spatial_shape: tuple[int, int], has_intent_tokens: bool) -> dict[str, np.ndarray]:
    """Keep tensors from one model call together; never join another pass by key."""
    from lead.tfv6.trajectory_scene_scorer import compress_planner_scene_tokens

    required = (
        prediction.pred_scene_tokens, prediction.pred_route_multimodal,
        prediction.pred_route_anchor, prediction.pred_route_conf,
        prediction.pred_route_selected_idx, prediction.pred_target_speed_scalar,
    )
    if any(value is None for value in required):
        raise RuntimeError("corridor model did not expose scene, paths, anchors, confidence and speed")
    original = prediction.pred_route_multimodal.float().cpu().numpy()
    anchor = prediction.pred_route_anchor.float().cpu().numpy()
    confidence = prediction.pred_route_conf.float().cpu().numpy()
    winner = prediction.pred_route_selected_idx.long().cpu().numpy()
    batch, path_count, points, coordinates = original.shape
    if coordinates != 2 or points < 2 or len(keys) != batch:
        raise ValueError("unexpected route shape or key count")
    if anchor.shape[:2] != (batch, path_count) or confidence.shape != (batch, path_count):
        raise ValueError("route/anchor/confidence shape mismatch")
    if ((winner < 0) | (winner >= path_count)).any():
        raise ValueError("selected route index out of range")
    original_valid = anchor[..., 3] > 0.5
    anchor_fallback = ~original_valid[np.arange(batch), winner]
    # The planner executes its fallback winner even when no anchor passed the
    # validity threshold. The raw candidate must remain part of the vocabulary.
    original_valid[np.arange(batch), winner] = True
    local = local_path_variants(
        original[np.arange(batch), winner], offsets, ramp_m=8.0,
    )
    routes = np.concatenate((original, local), axis=1)
    local_valid = np.broadcast_to(
        original_valid[np.arange(batch), winner, None], (batch, len(offsets))
    )
    scene = compress_planner_scene_tokens(
        prediction.pred_scene_tokens.float(), spatial_shape=spatial_shape,
        has_intent_tokens=has_intent_tokens,
    ).cpu().numpy()
    speed = data["speed"].float().reshape(-1).cpu().numpy()
    target = prediction.pred_target_speed_scalar.float().reshape(-1).cpu().numpy()
    expert = data["route"].float().cpu().numpy()
    if expert.shape != (batch, points, 2):
        raise ValueError(f"expert route shape mismatch: {expert.shape}")
    if not (np.isfinite(scene).all() and np.isfinite(routes).all()
            and np.isfinite(speed).all() and np.isfinite(target).all()):
        raise ValueError("non-finite scene, path or speed")
    return {
        "keys": np.asarray(keys),
        "scene_tokens": scene.astype(np.float16),
        "routes": routes.astype(np.float16),
        "route_valid": np.concatenate((original_valid, local_valid), axis=1),
        "route_conf": confidence.astype(np.float16),
        "winner": winner.astype(np.int8),
        "current_speed": speed.astype(np.float16),
        "raw_target_speed": target.astype(np.float16),
        "expert_route": expert.astype(np.float16),
        "command": data["command"].cpu().numpy().astype(np.int8),
        "anchor_fallback": anchor_fallback,
    }


def verify_shard(path: Path, checkpoint: Path, manifest: Path, nearest: Path,
                 start: int, end: int, offsets: tuple[float, ...]) -> int:
    with np.load(path, allow_pickle=False) as archive:
        if not set(FIELDS).issubset(archive.files):
            raise ValueError(f"incomplete joint scene shard: {path}")
        if int(archive["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(f"schema mismatch in {path}")
        if not bool(archive["same_forward"]):
            raise ValueError(f"same-forward provenance missing in {path}")
        if str(archive["source_checkpoint"]) != str(checkpoint.resolve()):
            raise ValueError(f"checkpoint mismatch in {path}")
        if int(archive["checkpoint_size"]) != checkpoint.stat().st_size or int(
            archive["checkpoint_mtime_ns"]
        ) != checkpoint.stat().st_mtime_ns:
            raise ValueError(f"checkpoint file changed since {path} was written")
        if str(archive["source_manifest"]) != str(manifest.resolve()) or str(
            archive["nearest_vlm_manifest"]
        ) != str(nearest.resolve()):
            raise ValueError(f"manifest mismatch in {path}")
        if int(archive["start"]) != start or int(archive["end"]) != end:
            raise ValueError(f"interval mismatch in {path}")
        if not np.allclose(archive["local_offsets_m"], offsets):
            raise ValueError(f"local offsets mismatch in {path}")
        keys = archive["keys"]
        if len(keys) != end - start or len(set(map(str, keys))) != len(keys):
            raise ValueError(f"frame count/uniqueness mismatch in {path}")
        for name in FIELDS:
            if len(archive[name]) != len(keys):
                raise ValueError(f"{name} row count mismatch in {path}")
        if archive["route_valid"].shape[1] != archive["routes"].shape[1]:
            raise ValueError(f"route validity shape mismatch in {path}")
        if archive["scene_tokens"].ndim != 3:
            raise ValueError(f"scene token shape mismatch in {path}")
        if archive["route_conf"].shape[1] + len(offsets) != archive["routes"].shape[1]:
            raise ValueError(f"joint Path count mismatch in {path}")
        winner = archive["winner"].astype(np.int64)
        if ((winner < 0) | (winner >= archive["route_conf"].shape[1])).any():
            raise ValueError(f"winner index out of range in {path}")
        if not archive["route_valid"][np.arange(len(keys)), winner].all():
            raise ValueError(f"executed winner is marked invalid in {path}")
    return len(keys)


def parse_intervals(start: int | None, end: int | None,
                    encoded: str | None) -> list[tuple[int, int]]:
    """Accept one legacy shard or a comma-separated list of disjoint shards."""
    if encoded is not None:
        if start is not None or end is not None:
            raise ValueError("--intervals cannot be combined with --start/--end")
        try:
            intervals = [tuple(map(int, item.split(":"))) for item in encoded.split(",")]
        except ValueError as error:
            raise ValueError("--intervals must be start:end[,start:end...]") from error
    else:
        if start is None or end is None:
            raise ValueError("provide --start/--end or --intervals")
        intervals = [(start, end)]
    if not intervals or any(len(item) != 2 or item[0] < 0 or item[1] <= item[0]
                            for item in intervals):
        raise ValueError("each interval must be a nonempty nonnegative start:end")
    intervals = sorted(intervals)
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("intervals overlap")
    return intervals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--ckpt-name", default="model_0019.pth")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nearest-vlm-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--intervals", help="start:end[,start:end...] on one GPU")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--local-offsets", default="-1.5,-0.75,0.75,1.5")
    args = parser.parse_args()
    offsets = tuple(float(value) for value in args.local_offsets.split(","))
    if len(offsets) != 4 or len(set(offsets)) != 4 or any(
        not np.isfinite(value) or abs(value) > 2 or value == 0 for value in offsets
    ):
        parser.error("expected four unique nonzero local offsets within +/-2 m")
    try:
        intervals = parse_intervals(args.start, args.end, args.intervals)
    except ValueError as error:
        parser.error(str(error))
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("invalid batch size or worker count")
    checkpoint = args.ckpt_dir / args.ckpt_name
    if not checkpoint.is_file():
        parser.error(f"missing checkpoint: {checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for start, end in intervals:
        output = args.output_dir / f"joint_scene_{start:06d}_{end:06d}.npz"
        if output.exists():
            count = verify_shard(
                output, checkpoint, args.manifest, args.nearest_vlm_manifest,
                start, end, offsets,
            )
            print(f"verified existing {output}: {count} frames; skipping")
        else:
            pending.append((start, end, output))
    if not pending:
        return

    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.tfv6.tfv6 import TFv6
    from lead.training.config_training import TrainingConfig

    with (args.ckpt_dir / "config.json").open() as handle:
        config = TrainingConfig(json.load(handle), raise_error_on_missing_key=False)
    if not config.multimodal_planner:
        raise ValueError("joint features require the multimodal corridor checkpoint")
    config.route_selection_mode = "confidence"
    config.route_speed_safety_gate = False
    config.route_future_safety_gate = False
    config.route_velocity_scorer_gate = False
    config.route_predicted_actor_velocity_gate = False
    config.use_sensor_perburtation = False
    import timm
    original_create = timm.create_model
    timm.create_model = lambda *a, **k: original_create(
        *a, **{**k, "pretrained": False}
    )
    if not torch.cuda.is_available():
        raise RuntimeError("B3b joint scene extraction requires a visible CUDA GPU")
    device = torch.device("cuda:0")
    model = TFv6(device, config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model.eval().requires_grad_(False)
    config.defer_vlm_inputs_to_wrapper = True
    config.detect_boxes = False
    config.use_semantic = False
    config._loaded_config["use_depth"] = False
    config.use_bev_semantic = False
    base = CARLAData(root=config.carla_data, config=config, random=False)
    dataset = VLMIntentDataset(
        base, vlm_cache_dir=config.vlm_cache_dir,
        manifest_path=str(args.manifest), anchor_cache_dir=None,
        nearest_cache_manifest_path=str(args.nearest_vlm_manifest),
    )
    if max(end for _, end, _ in pending) > len(dataset.valid_indices):
        raise ValueError(f"requested interval exceeds {len(dataset.valid_indices)} eligible frames")
    eligible_indices = dataset.valid_indices
    # A dense split can contain >870k entries. Keep only this worker's keys,
    # rather than replicating the entire manifest dictionary on every GPU.
    selected_pairs = set()
    selected_count = 0
    for start, end, _ in pending:
        for index in eligible_indices[start:end]:
            image_path = str(base.images[index], encoding="utf-8")
            parts = image_path.split("/")
            selected_pairs.add((parts[-3], parts[-1].split(".")[0]))
            selected_count += 1
    if len(selected_pairs) != selected_count:
        raise ValueError("duplicate route/frame pairs in selected dataset shard")
    key_by_route_frame = {}
    with args.manifest.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            pair = (entry["route"], entry["frame"])
            if pair in selected_pairs:
                if pair in key_by_route_frame:
                    raise ValueError(f"duplicate manifest entry: {pair}")
                key_by_route_frame[pair] = entry["key"]
    if len(key_by_route_frame) != len(selected_pairs):
        raise ValueError("manifest does not cover every selected dataset frame")
    for start, end, output in pending:
        dataset.valid_indices = eligible_indices[start:end]
        loader_kwargs = dict(
            dataset=dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=_collate,
            pin_memory=device.type == "cuda",
        )
        if args.num_workers:
            loader_kwargs["prefetch_factor"] = 1
        loader = DataLoader(**loader_kwargs)
        saved: dict[str, list[np.ndarray]] = {name: [] for name in FIELDS}
        data_wait_s = 0.0
        forward_pack_s = 0.0
        last_end = time.perf_counter()
        for data in tqdm(loader, desc=f"joint scene {start}:{end}", unit="batch"):
            now = time.perf_counter()
            data_wait_s += now - last_end
            data.pop("anchor", None)  # use anchors predicted by the frozen model
            with torch.inference_mode(), torch.amp.autocast(
                device_type=device.type, dtype=config.torch_float_type,
                enabled=config.use_mixed_precision_training and device.type == "cuda",
            ):
                prediction = model(data)
            keys = [
                key_by_route_frame[(route, frame)]
                for route, frame in zip(data["route_number"], data["frame_number"], strict=True)
            ]
            batch = pack_batch(
                prediction, data, keys, offsets,
                spatial_shape=(config.lidar_vert_anchors, config.lidar_horz_anchors),
                has_intent_tokens=config.use_control_conditioning,
            )
            for name in FIELDS:
                saved[name].append(batch[name])
            last_end = time.perf_counter()
            forward_pack_s += last_end - now
        packed = {name: np.concatenate(saved[name], axis=0) for name in FIELDS}
        if len(packed["keys"]) != end - start:
            raise RuntimeError("dataset frame count changed during extraction")
        packed.update(
            schema_version=np.asarray(SCHEMA_VERSION),
            same_forward=np.asarray(True),
            source_checkpoint=np.asarray(str(checkpoint.resolve())),
            checkpoint_size=np.asarray(checkpoint.stat().st_size),
            checkpoint_mtime_ns=np.asarray(checkpoint.stat().st_mtime_ns),
            source_manifest=np.asarray(str(args.manifest.resolve())),
            nearest_vlm_manifest=np.asarray(str(args.nearest_vlm_manifest.resolve())),
            start=np.asarray(start), end=np.asarray(end),
            original_path_count=np.asarray(packed["route_conf"].shape[1]),
            local_offsets_m=np.asarray(offsets, dtype=np.float32),
        )
        partial = output.with_name(output.name + ".partial.npz")
        write_start = time.perf_counter()
        np.savez_compressed(partial, **packed)
        verify_shard(
            partial, checkpoint, args.manifest, args.nearest_vlm_manifest,
            start, end, offsets,
        )
        os.replace(partial, output)
        print(f"wrote {output}: frames={len(packed['keys'])} "
              f"scene={packed['scene_tokens'].shape[1:]} paths={packed['routes'].shape[1:]} "
              f"data_wait={data_wait_s:.1f}s forward_pack={forward_pack_s:.1f}s "
              f"write_verify={time.perf_counter()-write_start:.1f}s")


if __name__ == "__main__":
    main()
