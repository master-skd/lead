"""P5b open-loop eval: measure the ACTUAL p5b_S / p5b_M model's imitation quality.

Unlike eval_p4b_openloop.py (a hot-swap ablation that feeds a VLM intent into the
*frozen P2* planner), this loads the real jointly-trained p5b checkpoint and runs
its canonical forward -- exactly reproducing training:

    backbone(sensors) + frozen VLMIntentDecoder(cached vlm_hidden) -> planner -> waypoints

Everything (config, vlm_cache_dir, vlm_manifest, vlm_intent_ckpt) is read from the
checkpoint's own config.json, so the same script evaluates S and M by just pointing
--ckpt-dir at the right folder. Frames are that config's VLM-cache subset (stride-10).

Why this exists: closed-loop p5b_S=72.9 / p5b_M=66.2 DS collapsed vs LEAD/P2 ~93.
This open-loop ADE/FDE on the TRAINING-CACHE features separates the two hypotheses:
    - ADE ~= P2 (0.185) here but closed-loop collapses -> NOT data/imitation; the
      break is closed-loop-specific (prime suspect: live IPC VLM features != cache).
    - ADE already bad here -> 1/10-data underfit is real; needs more data.

The optional zero-intent pass (--with-zero) feeds an all-zero intent to the same
model: if it matches the real-intent number, the planner barely uses intent even
after joint training.

Usage (single- or multi-GPU via torchrun):
    CUDA_VISIBLE_DEVICES=0 python scripts/p4/eval_p5b_openloop.py \
        --ckpt-dir outputs/local_training/p5b_S
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
        scripts/p4/eval_p5b_openloop.py --ckpt-dir outputs/local_training/p5b_M
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


def _errors(pred: torch.Tensor, label: torch.Tensor, common_utils) -> tuple[float, float]:
    """Return (ADE, FDE) for a predicted vs. ground-truth trajectory batch."""
    return (
        common_utils.average_displacement_error(pred, label),
        common_utils.final_displacement_error(pred, label),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True,
                    help="p5b checkpoint dir (e.g. outputs/local_training/p5b_S). "
                         "config.json, vlm_cache_dir, vlm_manifest, vlm_intent_ckpt "
                         "are all read from here.")
    ap.add_argument("--ckpt-name", default="model_0019.pth",
                    help="checkpoint filename inside --ckpt-dir")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="cap #frames for smoke test")
    ap.add_argument("--with-zero", action="store_true",
                    help="also eval with an all-zero intent (diagnostic: does the "
                         "planner actually use the intent condition?)")
    ap.add_argument("--out", default=None,
                    help="metrics json path (default: <ckpt-dir>/openloop_metrics.json)")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from lead.common import common_utils
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.tfv6.tfv6 import TFv6
    from lead.training.config_training import TrainingConfig
    from lead.training.mixed_training_utils import mixed_data_collate_fn

    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    out_path = args.out or os.path.join(args.ckpt_dir, "openloop_metrics.json")

    # --- DDP setup (torchrun sets RANK/WORLD_SIZE/LOCAL_RANK) ---
    # gloo backend: the only collective is a small all_reduce at the end, which NCCL
    # would route through /dev/shm (unavailable/too-small in this container). gloo
    # uses TCP on CPU tensors and sidesteps the shared-memory segment entirely.
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

    # --- Config: reconstruct the exact p5b config from the checkpoint's config.json ---
    with open(os.path.join(args.ckpt_dir, "config.json")) as f:
        loaded_config = json.load(f)
    config = TrainingConfig(loaded_config, raise_error_on_missing_key=False)
    # NB: config.device is a read-only @property derived from local_rank; it already
    # resolves to cuda:{local_rank}, matching `device`. TFv6 takes device as a ctor arg.

    # The ResNet image backbone builds with pretrained=True (timm fetches from HF hub);
    # pointless here since the full p5b checkpoint overwrites it, and impossible offline.
    # Force pretrained=False so construction never touches the network.
    import timm
    _timm_create_model = timm.create_model
    timm.create_model = lambda *a, **k: _timm_create_model(*a, **{**k, "pretrained": False})

    # --- Model: build TFv6 (loads the frozen VLM intent head from config at __init__),
    #     then load the jointly-trained p5b weights over everything. ---
    model = TFv6(device, config).to(device)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True),
        strict=True,
    )
    model.eval().requires_grad_(False)
    if is_main:
        print(f"Loaded p5b model from {ckpt_path}")
        print(f"  vlm_intent_ckpt = {config.vlm_intent_ckpt}")
        print(f"  vlm_cache_dir   = {config.vlm_cache_dir}")
        print(f"  vlm_manifest    = {config.vlm_manifest}")

    # --- Data: full-sensor CARLAData + per-frame VLM cache (the training subset) ---
    carla_ds = CARLAData(root=config.carla_data, config=config)
    vlm_ds = VLMIntentDataset(
        carla_ds,
        vlm_cache_dir=config.vlm_cache_dir,
        manifest_path=config.vlm_manifest,
    )
    sampler = (
        DistributedSampler(vlm_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if is_ddp
        else None
    )
    loader = DataLoader(
        vlm_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=mixed_data_collate_fn,  # keeps full sensors + vlm_hidden (NOT the
        pin_memory=True,                   # intent-only vlm_intent_collate_fn)
    )
    local_limit = None if args.limit is None else math.ceil(args.limit / world_size)

    n_wp = config.num_way_points_prediction

    keys = ["route_ade", "route_fde", "wp_ade", "wp_fde"]
    real = {k: 0.0 for k in keys}          # canonical p5b forward (VLM intent)
    zero = {k: 0.0 for k in keys}          # all-zero intent control (optional)
    n_seen = 0

    for data in tqdm(loader, desc="P5b open-loop eval", disable=not is_main):
        if local_limit is not None and n_seen >= local_limit:
            break
        bs = data["vlm_hidden"].shape[0]
        route_label = data["route"].float()
        wp_label = data["future_waypoints"].float()[:, :n_wp]

        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda",
            dtype=config.torch_float_type,
            enabled=config.use_mixed_precision_training,
        ):
            pred_real = model(data)  # canonical: intent from frozen VLM head on vlm_hidden
            pred_zero = None
            if args.with_zero:
                # external_intent overrides the VLM head with a zero field of the same shape.
                zero_intent = torch.zeros(
                    bs, 1, config.lidar_height_pixel, config.lidar_width_pixel,
                    device=device, dtype=torch.float32,
                )
                pred_zero = model(data, external_intent=zero_intent)

        r_ade, r_fde = _errors(pred_real.pred_route, route_label, common_utils)
        w_ade, w_fde = _errors(pred_real.pred_future_waypoints, wp_label, common_utils)
        real["route_ade"] += r_ade * bs
        real["route_fde"] += r_fde * bs
        real["wp_ade"] += w_ade * bs
        real["wp_fde"] += w_fde * bs

        if pred_zero is not None:
            zr_ade, zr_fde = _errors(pred_zero.pred_route, route_label, common_utils)
            zw_ade, zw_fde = _errors(pred_zero.pred_future_waypoints, wp_label, common_utils)
            zero["route_ade"] += zr_ade * bs
            zero["route_fde"] += zr_fde * bs
            zero["wp_ade"] += zw_ade * bs
            zero["wp_fde"] += zw_fde * bs

        n_seen += bs

    # --- Aggregate sums across ranks BEFORE normalizing (gloo -> CPU tensor) ---
    if is_ddp:
        packed = torch.tensor(
            [real[k] for k in keys] + [zero[k] for k in keys] + [float(n_seen)],
            device="cpu",
            dtype=torch.float64,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        vals = packed.tolist()
        real = {k: vals[i] for i, k in enumerate(keys)}
        zero = {k: vals[4 + i] for i, k in enumerate(keys)}
        n_seen = int(round(vals[8]))

    for acc in (real, zero):
        for k in keys:
            acc[k] /= max(n_seen, 1)

    if not is_main:
        dist.destroy_process_group()
        return

    # --- Report (rank 0 only) ---
    print(f"\n=== P5b open-loop eval: {args.ckpt_dir} ({n_seen} frames) ===")
    if args.with_zero:
        print(f"{'metric':<12}{'p5b(VLM intent)':>18}{'zero-intent':>14}{'Δ(zero-real)':>15}")
        for k in keys:
            print(f"{k:<12}{real[k]:>18.4f}{zero[k]:>14.4f}{zero[k]-real[k]:>+15.4f}")
        print("Read: zero≈real -> planner barely uses intent even after joint training.")
    else:
        print(f"{'metric':<12}{'p5b(VLM intent)':>18}")
        for k in keys:
            print(f"{k:<12}{real[k]:>18.4f}")
    print("\nCompare wp_ade to P2's open-loop 0.185: close -> imitation OK, closed-loop "
          "break is elsewhere (suspect live IPC features); much worse -> 1/10-data underfit.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "n_frames": n_seen,
                "ckpt": ckpt_path,
                "vlm_intent_ckpt": config.vlm_intent_ckpt,
                "vlm_cache_dir": config.vlm_cache_dir,
                "p5b_vlm_intent": real,
                **({"zero_intent_control": zero} if args.with_zero else {}),
            },
            f,
            indent=2,
        )
    print(f"\n✓ Wrote {out_path}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
