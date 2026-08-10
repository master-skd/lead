"""P5 B2 open-loop eval: the K-arm planner scored three ways, plus arm-divergence stats.

Why not reuse eval_p5_stepA_openloop.py: in multimodal mode PlanningDecoder sets
``route = route_all[:, 0]`` as an explicit placeholder, so ``pred.pred_route`` is ARM 0 --
index 0, with no reason to be the good arm. Its ADE says nothing about B2 vs B1'''.

So report all three, because they answer different questions:
  ade_conf   -- the arm the confidence head ranks first. This is the honest single-number
                comparison against B1''' (a single-route planner has exactly one answer, and
                B2 must also commit to one when there is no obstacle to break the tie).
  ade_oracle -- the best of K arms (min ADE). Upper bound: how much route diversity is
                actually available. oracle << conf means the arms are good but the
                confidence ranking is bad -- a fixable head, not a broken planner.
  ade_arm0   -- what the Step A script would have printed. Kept only so the two scripts'
                numbers can be reconciled; do NOT quote it as B2's score.

Divergence stats are the actual B2 go/no-go. WTA collapse (the DiffusionDrive failure the
anchor hinges exist to prevent) shows up as spread_endpoint -> 0: K arms that all trace the
same path. A healthy junction frame should show arms separated by metres.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
        scripts/p4/eval_p5_stepB2_openloop.py --ckpt-dir outputs/local_training/p5_stepB2_final
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
    p, g = pred[:, lo:hi], label[:, lo:hi]
    return (
        torch.linalg.norm(p - g, dim=-1).mean().item(),
        torch.linalg.norm(p[:, -1] - g[:, -1], dim=-1).mean().item(),
    )


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
    out_path = args.out or os.path.join(args.ckpt_dir, "openloop_b2_metrics.json")

    is_ddp = "RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend="gloo")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    with open(os.path.join(args.ckpt_dir, "config.json")) as f:
        config = TrainingConfig(json.load(f), raise_error_on_missing_key=False)
    assert config.multimodal_planner, (
        f"{args.ckpt_dir} is not a multimodal run -- use eval_p5_stepA_openloop.py"
    )

    import timm
    _create = timm.create_model
    timm.create_model = lambda *a, **k: _create(*a, **{**k, "pretrained": False})

    model = TFv6(device, config).to(device)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True), strict=True,
    )
    model.eval().requires_grad_(False)

    near = config.route_near_points or 10
    n_route = config.num_route_points_prediction
    if is_main:
        print(f"Loaded {ckpt_path}\n  K={config.multimodal_planner_k} "
              f"route points={n_route} near split={near}")

    carla_ds = CARLAData(root=config.carla_data, config=config)
    vlm_ds = VLMIntentDataset(
        carla_ds, vlm_cache_dir=config.vlm_cache_dir, manifest_path=config.vlm_manifest,
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

    keys = [
        "ade_conf_all", "ade_conf_near", "fde_conf_near", "ade_conf_far", "fde_conf_far",
        "ade_oracle_all", "ade_oracle_near", "ade_arm0_all",
        "spread_endpoint", "n_valid_arms", "conf_is_oracle",
        "multi_arm_frames", "spread_endpoint_multi",
    ]
    acc = {k: 0.0 for k in keys}
    n_seen = 0

    for data in tqdm(loader, desc="B2 open-loop", disable=not is_main):
        if local_limit is not None and n_seen >= local_limit:
            break
        bs = data["vlm_hidden"].shape[0]
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", dtype=config.torch_float_type,
            enabled=config.use_mixed_precision_training,
        ):
            model(data)  # multimodal outputs are bypassed via `data`, not the return value

        route_all = data["route_multimodal"].float()          # (B,K,n,2)
        conf = data["route_conf"].float()                     # (B,K)
        valid = data["anchor"].to(route_all.device).float()[:, :, 3]  # (B,K)
        gl = data["route"].float().to(route_all.device)       # (B,n,2)
        B, K = valid.shape

        # per-arm mean-L2 to GT; invalid arms are excluded from every selection
        d = torch.linalg.norm(route_all - gl[:, None], dim=-1).mean(dim=2)  # (B,K)
        big = (1.0 - valid) * 1e9
        idx_conf = (conf - big).argmax(dim=1)     # highest-confidence VALID arm
        idx_oracle = (d + big).argmin(dim=1)      # closest-to-GT VALID arm
        ar = torch.arange(B, device=route_all.device)

        for tag, sel in (("conf", route_all[ar, idx_conf]), ("oracle", route_all[ar, idx_oracle])):
            a_all, _ = _seg_ade_fde(sel, gl, 0, n_route)
            a_near, f_near = _seg_ade_fde(sel, gl, 0, near)
            acc[f"ade_{tag}_all"] += a_all * bs
            acc[f"ade_{tag}_near"] += a_near * bs
            if tag == "conf":
                a_far, f_far = _seg_ade_fde(sel, gl, near, n_route)
                acc["fde_conf_near"] += f_near * bs
                acc["ade_conf_far"] += a_far * bs
                acc["fde_conf_far"] += f_far * bs
        acc["ade_arm0_all"] += _seg_ade_fde(route_all[:, 0], gl, 0, n_route)[0] * bs

        # --- divergence: mean pairwise endpoint distance among VALID arms (metres).
        # This is the anti-collapse read-out; 0 means all arms traced the same path.
        ends = route_all[:, :, -1, :]                                   # (B,K,2)
        pdist = torch.linalg.norm(ends[:, :, None] - ends[:, None, :], dim=-1)  # (B,K,K)
        pair_m = valid[:, :, None] * valid[:, None, :]
        pair_m = pair_m * (1.0 - torch.eye(K, device=pair_m.device))    # drop self-pairs
        n_pair = pair_m.sum(dim=(1, 2))                                 # (B,)
        spread = (pdist * pair_m).sum(dim=(1, 2)) / n_pair.clamp(min=1.0)  # (B,)
        acc["spread_endpoint"] += spread.sum().item()
        acc["n_valid_arms"] += valid.sum().item()
        acc["conf_is_oracle"] += (idx_conf == idx_oracle).float().sum().item()
        # Single-arm frames (80% of the dataset) have no pairs and would drag the mean to 0,
        # so track the genuinely-forking frames separately -- that is where B2 must diverge.
        multi = (n_pair > 0).float()
        acc["multi_arm_frames"] += multi.sum().item()
        acc["spread_endpoint_multi"] += (spread * multi).sum().item()
        n_seen += bs

    if is_ddp:
        packed = torch.tensor([acc[k] for k in keys] + [float(n_seen)], dtype=torch.float64)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        vals = packed.tolist()
        acc = {k: vals[i] for i, k in enumerate(keys)}
        n_seen = int(round(vals[len(keys)]))

    n_multi = acc["multi_arm_frames"]
    for k in keys:
        if k == "spread_endpoint_multi":
            acc[k] /= max(n_multi, 1.0)   # average over forking frames only
        elif k != "multi_arm_frames":
            acc[k] /= max(n_seen, 1)

    if not is_main:
        dist.destroy_process_group()
        return

    print(f"\n=== B2 open-loop: {args.ckpt_dir} ({n_seen} frames) ===")
    print(f"  conf-selected   ALL={acc['ade_conf_all']:.4f}  "
          f"NEAR ADE={acc['ade_conf_near']:.4f} FDE={acc['fde_conf_near']:.4f}  "
          f"FAR ADE={acc['ade_conf_far']:.4f} FDE={acc['fde_conf_far']:.4f}")
    print(f"  oracle (best-K) ALL={acc['ade_oracle_all']:.4f}  NEAR={acc['ade_oracle_near']:.4f}")
    print(f"  arm0 placeholder ALL={acc['ade_arm0_all']:.4f}  (not B2's score, for reconciliation only)")
    print(f"\n  arms/frame={acc['n_valid_arms']:.2f}  conf==oracle {acc['conf_is_oracle']*100:.1f}%")
    print(f"  endpoint spread: {acc['spread_endpoint']:.2f} m all frames | "
          f"{acc['spread_endpoint_multi']:.2f} m over {int(n_multi)} forking frames")
    print("\nGo/no-go: compare ade_conf_near against B1''' 0.0375 (the single-route planner "
          "it was initialised from). spread_endpoint_multi ~0 == WTA collapse.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"n_frames": n_seen, "ckpt": ckpt_path, "near_split": near,
                   "n_route": n_route, "K": config.multimodal_planner_k,
                   "n_forking_frames": int(n_multi), **acc}, f, indent=2)
    print(f"\n✓ Wrote {out_path}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
