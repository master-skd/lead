"""P5 Step A open-loop eval: near/far route ADE for the extended 30m route head.

The extended route head predicts 30 points (~30m). Closed-loop steering only consumes
route[:~8m] (PID n_lookahead in [0,8]), so the go/no-go for Step A is:
    - NEAR segment (first `route_near_points`, ~10m) ADE ~= P2's ~0.030 -> the extension
      did NOT damage the control-critical near path (closed-loop safety preserved).
    - FAR segment (10-30m) ADE is allowed to be higher -> it's the fuzzy intent horizon,
      the substrate the multimodal arms (Step B) will diverge on.

Reuses the p5b canonical forward (backbone + frozen VLM intent head -> planner). All paths
read from the checkpoint's own config.json.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
        scripts/p4/eval_p5_stepA_openloop.py --ckpt-dir outputs/local_training/p5_stepA
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def _seg_ade_fde(pred, label, lo, hi):
    """Mean per-point L2 (ADE) over point range [lo, hi), and L2 at the last point (FDE)."""
    p = pred[:, lo:hi]
    g = label[:, lo:hi]
    ade = torch.linalg.norm(p - g, dim=-1).mean().item()
    fde = torch.linalg.norm(p[:, -1] - g[:, -1], dim=-1).mean().item()
    return ade, fde


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--ckpt-name", default="model_0019.pth")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.tfv6.tfv6 import TFv6
    from lead.training.config_training import TrainingConfig
    from lead.training.mixed_training_utils import mixed_data_collate_fn

    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    out_path = args.out or os.path.join(args.ckpt_dir, "openloop_stepA_metrics.json")

    is_ddp = "RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend="gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    with open(os.path.join(args.ckpt_dir, "config.json")) as f:
        loaded_config = json.load(f)
    config = TrainingConfig(loaded_config, raise_error_on_missing_key=False)

    import timm
    _timm_create_model = timm.create_model
    timm.create_model = lambda *a, **k: _timm_create_model(*a, **{**k, "pretrained": False})

    model = TFv6(device, config).to(device)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True),
        strict=True,
    )
    model.eval().requires_grad_(False)

    near = config.route_near_points or 10
    n_route = config.num_route_points_prediction
    if is_main:
        print(f"Loaded {ckpt_path}")
        print(f"  route points = {n_route}, near split = {near}")

    carla_ds = CARLAData(root=config.carla_data, config=config)
    vlm_ds = VLMIntentDataset(
        carla_ds,
        vlm_cache_dir=config.vlm_cache_dir,
        manifest_path=config.vlm_manifest,
    )
    sampler = (
        DistributedSampler(vlm_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if is_ddp else None
    )
    loader = DataLoader(
        vlm_ds, batch_size=args.batch_size, shuffle=False, sampler=sampler,
        num_workers=args.num_workers, collate_fn=mixed_data_collate_fn, pin_memory=True,
    )
    local_limit = None if args.limit is None else math.ceil(args.limit / world_size)

    keys = ["route_ade_all", "route_ade_near", "route_fde_near",
            "route_ade_far", "route_fde_far"]
    acc = {k: 0.0 for k in keys}
    n_seen = 0

    for data in tqdm(loader, desc="Step A open-loop", disable=not is_main):
        if local_limit is not None and n_seen >= local_limit:
            break
        bs = data["vlm_hidden"].shape[0]
        route_label = data["route"].float()
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", dtype=config.torch_float_type,
            enabled=config.use_mixed_precision_training,
        ):
            pred = model(data)
        pr = pred.pred_route.float()
        gl = route_label.to(pr.device)
        ade_all, _ = _seg_ade_fde(pr, gl, 0, n_route)
        ade_n, fde_n = _seg_ade_fde(pr, gl, 0, near)
        ade_f, fde_f = _seg_ade_fde(pr, gl, near, n_route)
        acc["route_ade_all"] += ade_all * bs
        acc["route_ade_near"] += ade_n * bs
        acc["route_fde_near"] += fde_n * bs
        acc["route_ade_far"] += ade_f * bs
        acc["route_fde_far"] += fde_f * bs
        n_seen += bs

    if is_ddp:
        packed = torch.tensor([acc[k] for k in keys] + [float(n_seen)],
                              device="cpu", dtype=torch.float64)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        vals = packed.tolist()
        acc = {k: vals[i] for i, k in enumerate(keys)}
        n_seen = int(round(vals[len(keys)]))

    for k in keys:
        acc[k] /= max(n_seen, 1)

    if not is_main:
        dist.destroy_process_group()
        return

    print(f"\n=== Step A open-loop: {args.ckpt_dir} ({n_seen} frames) ===")
    print(f"  route ADE (all {n_route} pts) : {acc['route_ade_all']:.4f}")
    print(f"  NEAR (0..{near}, ~10m)  ADE={acc['route_ade_near']:.4f}  FDE={acc['route_fde_near']:.4f}")
    print(f"  FAR  ({near}..{n_route}, ~30m) ADE={acc['route_ade_far']:.4f}  FDE={acc['route_fde_far']:.4f}")
    print("\nGo/no-go: NEAR ADE ~= P2's 0.030 -> extension preserved the control-critical "
          "near path; FAR is the intent horizon (higher OK).")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"n_frames": n_seen, "ckpt": ckpt_path, "near_split": near,
                   "n_route": n_route, **acc}, f, indent=2)
    print(f"\n✓ Wrote {out_path}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
