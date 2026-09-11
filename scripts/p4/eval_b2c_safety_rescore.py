"""Validate B2c safety gating and sweep its two thresholds without retraining.

By default cached lane-graph anchors are removed before inference, forcing the same
predicted-intent anchor extraction used in closed loop. One forward pass produces every
arm's fixed collision/off-corridor cost; all threshold pairs are then evaluated offline.

Example:
  CUDA_VISIBLE_DEVICES=0 python scripts/p4/eval_b2c_safety_rescore.py \
    --ckpt-dir outputs/local_training/p5_stepB2_corridor --limit 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def _floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _empty_stats() -> dict[str, float]:
    return {
        "n": 0.0,
        "baseline_unsafe": 0.0,
        "safe_alternate": 0.0,
        "switched": 0.0,
        "fallback": 0.0,
        "collision_before": 0.0,
        "collision_after": 0.0,
        "off_corridor_before": 0.0,
        "off_corridor_after": 0.0,
        "ade_before": 0.0,
        "ade_after": 0.0,
        "ade_near_before": 0.0,
        "ade_near_after": 0.0,
    }


def _add_stats(stats, mask, selection, collision, corridor, routes, label, near):
    if not mask.any():
        return
    ar = torch.arange(routes.shape[0], device=routes.device)
    base, picked = selection.baseline_idx, selection.selected_idx
    base_route, picked_route = routes[ar, base], routes[ar, picked]
    base_safe = selection.safe_mask[ar, base]
    ade_base = torch.linalg.norm(base_route - label, dim=-1)
    ade_picked = torch.linalg.norm(picked_route - label, dim=-1)
    stats["n"] += mask.sum().item()
    stats["baseline_unsafe"] += ((~base_safe) & mask).sum().item()
    stats["safe_alternate"] += (
        (~base_safe) & selection.safe_mask.any(dim=1) & mask
    ).sum().item()
    stats["switched"] += (selection.switched & mask).sum().item()
    stats["fallback"] += (selection.fallback & mask).sum().item()
    for key, values in (
        ("collision_before", collision[ar, base]),
        ("collision_after", collision[ar, picked]),
        ("off_corridor_before", corridor[ar, base]),
        ("off_corridor_after", corridor[ar, picked]),
        ("ade_before", ade_base.mean(dim=1)),
        ("ade_after", ade_picked.mean(dim=1)),
        ("ade_near_before", ade_base[:, :near].mean(dim=1)),
        ("ade_near_after", ade_picked[:, :near].mean(dim=1)),
    ):
        stats[key] += values[mask].sum().item()


def _finish(stats: dict[str, float]) -> dict[str, float | int]:
    n = max(stats["n"], 1.0)
    unsafe = max(stats["baseline_unsafe"], 1.0)
    out: dict[str, float | int] = {"n": int(stats["n"])}
    for key in ("baseline_unsafe", "switched", "fallback"):
        out[f"{key}_rate"] = stats[key] / n
    out["safe_alternate_rate"] = stats["safe_alternate"] / n
    out["rescue_rate_given_unsafe"] = stats["safe_alternate"] / unsafe
    for key in (
        "collision_before", "collision_after", "off_corridor_before",
        "off_corridor_after", "ade_before", "ade_after",
        "ade_near_before", "ade_near_after",
    ):
        out[key] = stats[key] / n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--ckpt-name", default="model_0019.pth")
    ap.add_argument("--anchor-source", choices=("predicted", "cached"), default="predicted")
    # max-near collision scores can legitimately occupy the upper half of [0,1], so
    # include broad values by default instead of producing an all-fallback sweep.
    ap.add_argument("--collision-thresholds", default="0.05,0.1,0.2,0.5,0.9")
    ap.add_argument("--corridor-thresholds", default="0.02,0.05,0.1,0.2")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.tfv6.route_safety_rescorer import select_indices_from_costs
    from lead.tfv6.tfv6 import TFv6
    from lead.training.config_training import TrainingConfig
    from lead.training.mixed_training_utils import mixed_data_collate_fn

    ckpt = os.path.join(args.ckpt_dir, args.ckpt_name)
    out_path = args.out or os.path.join(args.ckpt_dir, "b2c_safety_rescore.json")
    with open(os.path.join(args.ckpt_dir, "config.json")) as f:
        config = TrainingConfig(json.load(f), raise_error_on_missing_key=False)
    if not config.multimodal_planner:
        raise ValueError("B2c requires a multimodal_planner checkpoint")
    # One configured safety pass exposes per-arm costs; the sweep below does not rerun the net.
    config.route_selection_mode = "safety_rescore"

    import timm
    create_model = timm.create_model
    timm.create_model = lambda *a, **k: create_model(*a, **{**k, "pretrained": False})
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # The backbone explicitly casts inputs to config.torch_float_type. CPU convolutions
    # here use fp32 weights and cannot consume the checkpoint's default bf16 inputs.
    if device.type == "cpu":
        config.use_mixed_precision_training = False
    model = TFv6(device, config).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=True)
    model.eval().requires_grad_(False)

    base = CARLAData(root=config.carla_data, config=config)
    dataset = VLMIntentDataset(
        base, vlm_cache_dir=config.vlm_cache_dir, manifest_path=config.vlm_manifest,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=mixed_data_collate_fn, pin_memory=True,
    )
    collision_thresholds = _floats(args.collision_thresholds)
    corridor_thresholds = _floats(args.corridor_thresholds)
    pairs = [(c, o) for c in collision_thresholds for o in corridor_thresholds]
    accum = {f"{c:g},{o:g}": {"all": _empty_stats(), "multi": _empty_stats()} for c, o in pairs}
    near = max(1, min(int(config.route_safety_near_points), config.num_route_points_prediction))
    n_seen = 0

    for data in tqdm(loader, desc=f"B2c ({args.anchor_source} anchors)"):
        if args.limit is not None and n_seen >= args.limit:
            break
        if args.anchor_source == "predicted":
            data.pop("anchor", None)
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type,
            dtype=config.torch_float_type,
            enabled=config.use_mixed_precision_training and device.type == "cuda",
        ):
            pred = model(data)
        routes = pred.pred_route_multimodal.float()
        confidence = pred.pred_route_conf.float()
        anchors = pred.pred_route_anchor
        valid = (
            anchors[..., 3].to(confidence.device) > 0.5
            if anchors is not None else torch.ones_like(confidence, dtype=torch.bool)
        )
        collision = pred.pred_route_collision_cost.float()
        corridor = pred.pred_route_off_corridor_cost.float()
        label = data["route"].to(routes.device).float()
        bs = routes.shape[0]
        if args.limit is not None and n_seen + bs > args.limit:
            keep = args.limit - n_seen
            routes, confidence, valid = routes[:keep], confidence[:keep], valid[:keep]
            collision, corridor, label = collision[:keep], corridor[:keep], label[:keep]
            bs = keep
        multi = valid.sum(dim=1) > 1
        all_frames = torch.ones(bs, device=routes.device, dtype=torch.bool)

        for c_thr, o_thr in pairs:
            selection = select_indices_from_costs(
                confidence, valid, collision, corridor, c_thr, o_thr,
            )
            bucket = accum[f"{c_thr:g},{o_thr:g}"]
            _add_stats(bucket["all"], all_frames, selection, collision, corridor, routes, label, near)
            _add_stats(bucket["multi"], multi, selection, collision, corridor, routes, label, near)
        n_seen += bs

    results = {
        key: {scope: _finish(stats) for scope, stats in value.items()}
        for key, value in accum.items()
    }
    payload = {
        "checkpoint": ckpt,
        "anchor_source": args.anchor_source,
        "n_frames": n_seen,
        "near_points": near,
        "threshold_key": "collision,corridor",
        "results": results,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nB2c threshold sweep ({n_seen} frames; {args.anchor_source} anchors)")
    print(" coll  corr | multi rescue  switch fallback | coll before->after  off before->after  ADE before->after")
    for c_thr, o_thr in pairs:
        m = results[f"{c_thr:g},{o_thr:g}"]["multi"]
        print(
            f" {c_thr:4.2f}  {o_thr:4.2f} | {m['rescue_rate_given_unsafe']*100:5.1f}% "
            f"{m['switched_rate']*100:5.1f}% {m['fallback_rate']*100:5.1f}% | "
            f"{m['collision_before']:.3f}->{m['collision_after']:.3f}  "
            f"{m['off_corridor_before']:.3f}->{m['off_corridor_after']:.3f}  "
            f"{m['ade_before']:.3f}->{m['ade_after']:.3f}"
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
