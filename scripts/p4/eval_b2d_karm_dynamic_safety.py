"""Audit every predicted route arm against privileged future actor trajectories."""

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


_AUDIT_BATCH_KEYS = {
    "rgb", "rasterized_lidar", "vlm_hidden", "radar", "speed", "command",
    "target_point", "target_point_previous", "target_point_next", "route",
    "route_number", "frame_number", "anchor",
}


def _audit_collate(batch: list[dict]) -> dict:
    """Drop large training-only labels before worker-to-main shared-memory transfer."""

    slim = [{key: value for key, value in sample.items() if key in _AUDIT_BATCH_KEYS} for sample in batch]
    return torch.utils.data._utils.collate.default_collate(slim)


def _safe_div(numerator: np.ndarray | int, denominator: np.ndarray | int) -> float:
    return float(numerator / max(int(denominator), 1))


def _binary_auroc(score: np.ndarray, label: np.ndarray) -> float:
    score = np.asarray(score)
    label = np.asarray(label, dtype=bool)
    positives, negatives = int(label.sum()), int((~label).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(len(score), dtype=np.float64)
    # Pairwise rank form; ties are rare for model logits and do not affect the audit conclusion.
    return float((negatives * positives - (ranks[label].sum() - positives * (positives - 1) / 2)) / (positives * negatives))


def _format_optional(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def summarize_karm(
    confidence_logits: np.ndarray,
    valid: np.ndarray,
    collision: np.ndarray,
    ttc_s: np.ndarray,
    ade: np.ndarray,
) -> dict:
    """Summarize whether route confidence and dynamic safety carry different information."""

    n, k = valid.shape
    ar = np.arange(n)
    masked_conf = np.where(valid, confidence_logits, -np.inf)
    base = masked_conf.argmax(axis=1)
    masked_ade = np.where(valid, ade, np.inf)
    wta = masked_ade.argmin(axis=1)
    safe = valid & ~collision
    safe_conf = np.where(safe, confidence_logits, -np.inf)
    safe_pick = safe_conf.argmax(axis=1)
    has_safe = safe.any(axis=1)
    safe_pick = np.where(has_safe, safe_pick, base)
    base_unsafe = collision[ar, base]
    rescue = base_unsafe & has_safe
    arm_index = np.arange(k)[None, :]
    nonexpert = valid & (arm_index != wta[:, None])
    safe_nonexpert = nonexpert & ~collision
    probs = 1.0 / (1.0 + np.exp(np.clip(-confidence_logits, -60.0, 60.0)))
    valid_flat = valid.reshape(-1)
    collision_ttc = ttc_s[valid & collision]

    result = {
        "n": n,
        "multi_rate": float((valid.sum(axis=1) > 1).mean()),
        "mean_valid_arms": float(valid.sum() / max(n, 1)),
        "valid_arm_collision_rate": float(collision[valid].mean()),
        "confidence_winner_collision_rate": float(base_unsafe.mean()),
        "expert_wta_collision_rate": float(collision[ar, wta].mean()),
        "all_valid_arms_unsafe_rate": float((~has_safe).mean()),
        "unsafe_winner_with_safe_alternate_rate": float(rescue.mean()),
        "rescue_rate_given_unsafe_winner": _safe_div(rescue.sum(), base_unsafe.sum()),
        "oracle_safety_switch_rate": float((safe_pick != base).mean()),
        "confidence_winner_is_expert_wta_rate": float((base == wta).mean()),
        "ade_confidence_winner": float(ade[ar, base].mean()),
        "ade_expert_wta": float(ade[ar, wta].mean()),
        "ade_oracle_safe_confidence": float(ade[ar, safe_pick].mean()),
        "nonexpert_arm_safe_rate": float((~collision[nonexpert]).mean()) if nonexpert.any() else None,
        "confidence_safe_arm_auroc": _binary_auroc(
            probs.reshape(-1)[valid_flat], (~collision).reshape(-1)[valid_flat],
        ),
        "confidence_expert_wta_auroc": _binary_auroc(
            probs.reshape(-1)[valid_flat],
            (arm_index == wta[:, None]).reshape(-1)[valid_flat],
        ),
        "mean_confidence_probability": {
            "safe_valid": float(probs[safe].mean()) if safe.any() else None,
            "unsafe_valid": float(probs[valid & collision].mean()) if (valid & collision).any() else None,
            "safe_nonexpert": float(probs[safe_nonexpert].mean()) if safe_nonexpert.any() else None,
        },
        "safe_nonexpert_below_confidence": {
            f"{threshold:g}": float((probs[safe_nonexpert] < threshold).mean())
            if safe_nonexpert.any() else None
            for threshold in (0.05, 0.1, 0.25, 0.5)
        },
        "collision_ttc_s": {
            "count": int(len(collision_ttc)),
            "p10": float(np.quantile(collision_ttc, 0.1)) if len(collision_ttc) else None,
            "p50": float(np.quantile(collision_ttc, 0.5)) if len(collision_ttc) else None,
            "p90": float(np.quantile(collision_ttc, 0.9)) if len(collision_ttc) else None,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--ckpt-name", default="model_0019.pth")
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--future-cache-dir", required=True)
    parser.add_argument("--anchor-source", choices=("predicted", "cached"), default="predicted")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--safety-margin", type=float, default=0.2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.future_actor_cache import unpack_future_actor_frame
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.tfv6.future_collision import route_future_collision_label
    from lead.tfv6.tfv6 import TFv6
    from lead.training.config_training import TrainingConfig

    checkpoint = os.path.join(args.ckpt_dir, args.ckpt_name)
    output = args.out or os.path.join(args.ckpt_dir, "b2d_karm_dynamic_safety.json")
    with open(os.path.join(args.ckpt_dir, "config.json")) as handle:
        config = TrainingConfig(json.load(handle), raise_error_on_missing_key=False)
    if not config.multimodal_planner:
        raise ValueError("K-arm dynamic safety audit requires multimodal_planner")
    config.route_selection_mode = "confidence"
    config.route_speed_safety_gate = False
    config.use_sensor_perburtation = False

    with open(args.heldout_manifest) as handle:
        heldout_entries = [json.loads(line) for line in handle if line.strip()]
    entry_by_route_frame = {}
    for entry in heldout_entries:
        route_frame = (entry["route"], entry["frame"])
        if route_frame in entry_by_route_frame:
            raise ValueError(f"duplicate route/frame in heldout manifest: {route_frame}")
        entry_by_route_frame[route_frame] = entry

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
        raise FileNotFoundError(f"no heldout future cache shards in {args.future_cache_dir}")

    import timm
    create_model = timm.create_model
    timm.create_model = lambda *a, **k: create_model(*a, **{**k, "pretrained": False})
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        config.use_mixed_precision_training = False
    model = TFv6(device, config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model.eval().requires_grad_(False)

    # These auxiliary heads run after planning and cannot affect route/speed outputs. Skipping
    # them makes the audit materially faster and also prevents their large GT labels from being
    # loaded/collated. Radar detection stays enabled because its features condition this planner.
    config.detect_boxes = False
    config.use_semantic = False
    config._loaded_config["use_depth"] = False
    config.use_bev_semantic = False

    base = CARLAData(root=config.carla_data, config=config, random=False)
    dataset = VLMIntentDataset(
        base, vlm_cache_dir=config.vlm_cache_dir, manifest_path=args.heldout_manifest,
        anchor_cache_dir=(config.anchor_cache_dir if args.anchor_source == "cached" else None),
    )
    dataset.valid_indices = [
        index
        for index in dataset.valid_indices
        if (
            str(base.images[index], encoding="utf-8").split("/")[-3],
            str(base.images[index], encoding="utf-8").split("/")[-1].split(".")[0],
        ) in cached_route_frames
    ]
    if args.limit is not None:
        dataset.valid_indices = dataset.valid_indices[: args.limit]
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": _audit_collate,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        # A complete CARLA sample is large; the default prefetch factor can exhaust the
        # 512 MB /dev/shm found on the evaluation container.
        loader_kwargs["prefetch_factor"] = 1
    loader = DataLoader(**loader_kwargs)

    saved = {name: [] for name in ("confidence", "valid", "collision", "ttc", "ade")}
    n_seen = 0
    for data in tqdm(loader, desc=f"B2d K-arm ({args.anchor_source} anchors)"):
        if args.anchor_source == "predicted":
            data.pop("anchor", None)
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type,
            dtype=config.torch_float_type,
            enabled=config.use_mixed_precision_training and device.type == "cuda",
        ):
            prediction = model(data)
        routes = prediction.pred_route_multimodal.float().detach().cpu().numpy()
        confidence = prediction.pred_route_conf.float().detach().cpu().numpy()
        anchors = prediction.pred_route_anchor
        valid = (
            anchors[..., 3].detach().cpu().numpy() > 0.5
            if anchors is not None else np.ones_like(confidence, dtype=bool)
        )
        # Match inference fallback: if intent extraction produced only padding, slot 0
        # is still the deterministic route consumed by the controller.
        all_padding = ~valid.any(axis=1)
        valid[all_padding, 0] = True
        labels = data["route"].float().cpu().numpy()
        ade = np.linalg.norm(routes - labels[:, None], axis=-1).mean(axis=-1)
        raw_speed = prediction.pred_target_speed_scalar.float().reshape(-1)
        brake_probability = prediction.pred_target_speed_distribution.float().softmax(dim=1)[:, 0]
        raw_speed = torch.where(brake_probability > 0.9, torch.zeros_like(raw_speed), raw_speed)
        target_speeds = raw_speed.detach().cpu().numpy()
        current_speeds = data["speed"].float().reshape(-1).cpu().numpy()

        collision = np.zeros_like(valid, dtype=bool)
        ttc = np.full_like(confidence, np.inf, dtype=np.float32)
        for batch_index, (route_name, frame_name) in enumerate(
            zip(data["route_number"], data["frame_number"], strict=True),
        ):
            entry = entry_by_route_frame[(route_name, frame_name)]
            shard_path, local_index = cache_key_location[entry["key"]]
            actors = unpack_future_actor_frame(shards[shard_path], local_index)
            for arm_index in np.flatnonzero(valid[batch_index]):
                dynamic = route_future_collision_label(
                    routes[batch_index, arm_index],
                    float(current_speeds[batch_index]),
                    float(target_speeds[batch_index]),
                    actors,
                    safety_margin_m=args.safety_margin,
                    include_class_ids=(1, 2),
                )
                collision[batch_index, arm_index] = dynamic.collision
                ttc[batch_index, arm_index] = dynamic.ttc_s
        for name, value in (
            ("confidence", confidence), ("valid", valid), ("collision", collision),
            ("ttc", ttc), ("ade", ade),
        ):
            saved[name].append(value)
        n_seen += len(routes)

    values = {name: np.concatenate(parts) for name, parts in saved.items()}
    multi = values["valid"].sum(axis=1) > 1
    summaries = {"all": summarize_karm(**{
        "confidence_logits": values["confidence"], "valid": values["valid"],
        "collision": values["collision"], "ttc_s": values["ttc"], "ade": values["ade"],
    })}
    if multi.any():
        summaries["multi"] = summarize_karm(
            values["confidence"][multi], values["valid"][multi],
            values["collision"][multi], values["ttc"][multi], values["ade"][multi],
        )
    payload = {
        "checkpoint": checkpoint,
        "heldout_manifest": args.heldout_manifest,
        "future_cache_dir": args.future_cache_dir,
        "anchor_source": args.anchor_source,
        "n_frames": n_seen,
        "safety_margin_m": args.safety_margin,
        "dynamic_classes": ["car", "walker"],
        "summaries": summaries,
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nB2d K-arm dynamic safety ({n_seen} held-out expert frames)")
    for scope, result in summaries.items():
        print(
            f"[{scope}] n={result['n']} multi={result['multi_rate']:.1%} "
            f"arm-coll={result['valid_arm_collision_rate']:.1%} "
            f"winner-coll={result['confidence_winner_collision_rate']:.1%} "
            f"rescue={result['rescue_rate_given_unsafe_winner']:.1%} "
            f"safe-nonexpert={result['nonexpert_arm_safe_rate']:.1%} "
            f"safe-AUROC={result['confidence_safe_arm_auroc']:.3f}"
        )
        print(
            "  conf p safe/unsafe/safe-nonexpert="
            f"{_format_optional(result['mean_confidence_probability']['safe_valid'])}/"
            f"{_format_optional(result['mean_confidence_probability']['unsafe_valid'])}/"
            f"{_format_optional(result['mean_confidence_probability']['safe_nonexpert'])} "
            f"ADE base->safe={result['ade_confidence_winner']:.3f}->"
            f"{result['ade_oracle_safe_confidence']:.3f}"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
