"""P4a training script: train VLMIntentDecoder to distill expert route intent from Qwen-VL features.

Multi-GPU via torchrun DDP (only trains lightweight decoder, no backbone forward -> fast).

Usage (8 GPUs):
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/p4/train_vlm_intent_ddp.sh
Or directly:
    torchrun --standalone --nproc_per_node=8 scripts/p4/train_vlm_intent.py \
        --vlm-cache data/p4/vlm_cache --epochs 15 --batch-size 128 --lr 3e-4 \
        --logdir outputs/local_training/vlm_intent_p4a
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm-cache", default="data/p4/vlm_cache")
    ap.add_argument("--manifest", default=None,
                    help="extraction manifest; if given, cache membership is tested in "
                         "memory instead of one os.path.exists per frame (avoids a stat storm)")
    ap.add_argument("--val-manifest", default=None,
                    help="optional route-disjoint heldout manifest for validation")
    ap.add_argument("--split-metadata", default=None,
                    help="assert the dataset matches the route-disjoint split counts")
    ap.add_argument("--logdir", default="outputs/local_training/vlm_intent_p4a")
    ap.add_argument("--batch-size", type=int, default=128, help="GLOBAL batch size (split across GPUs)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    ap.add_argument("--multimodal-intent", action="store_true",
                    help="P5a: distill the multimodal drivable-support field (all "
                         "reachable arms) instead of the single expert route")
    ap.add_argument("--tversky-weight", type=float, default=0.0,
                    help="P5: weight of the recall-weighted Tversky term (0 = BCE only)")
    ap.add_argument("--lanegraph-label-dir", type=str, default=None,
                    help="P5+: read clean lane-graph corridor labels from this cache "
                         "instead of the mushy flood-fill blob")
    args = ap.parse_args()

    # --- DDP setup ---
    is_ddp = "RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if is_main:
        os.makedirs(args.logdir, exist_ok=True)

    # Config + dataset
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset, vlm_intent_collate_fn
    from lead.training.config_training import TrainingConfig
    from lead.tfv6.vlm_intent_decoder import VLMIntentDecoder

    config = TrainingConfig()
    config.use_planning_decoder = True
    config.use_intent_decoder = True
    # P5a: distill the multimodal drivable-support field instead of the expert route.
    config.use_multimodal_intent = args.multimodal_intent
    config.intent_tversky_weight = args.tversky_weight
    config.lanegraph_label_dir = args.lanegraph_label_dir
    # "plant" model_type makes CARLAData.__getitem__ return right after building the
    # visual_intent_label, skipping the heavy sensor path (lidar .laz). P4a only needs
    # the label, so this makes the dataloader fast.
    config.model_type = "plant"

    carla_ds = CARLAData(root=config.carla_data, config=config)
    vlm_ds = VLMIntentDataset(carla_ds, vlm_cache_dir=args.vlm_cache, manifest_path=args.manifest)
    val_ds = (
        VLMIntentDataset(carla_ds, vlm_cache_dir=args.vlm_cache, manifest_path=args.val_manifest)
        if args.val_manifest else None
    )
    if len(vlm_ds) == 0 or (val_ds is not None and len(val_ds) == 0):
        raise ValueError("train or heldout manifest has no frames in CARLAData")
    if args.split_metadata:
        with open(args.split_metadata) as stream:
            split = json.load(stream)
        if len(vlm_ds) != split["train_frames"] or val_ds is None or len(val_ds) != split["heldout_frames"]:
            raise ValueError(
                f"intent dataset/split mismatch: train={len(vlm_ds)}/{split['train_frames']} "
                f"heldout={len(val_ds) if val_ds is not None else 0}/{split['heldout_frames']}"
            )

    per_gpu_bs = max(1, args.batch_size // world_size)
    sampler = DistributedSampler(vlm_ds, shuffle=True) if is_ddp else None
    loader = DataLoader(
        vlm_ds,
        batch_size=per_gpu_bs,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=vlm_intent_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_sampler = (
        DistributedSampler(val_ds, shuffle=False) if is_ddp and val_ds is not None else None
    )
    val_loader = (
        DataLoader(
            val_ds, batch_size=per_gpu_bs, shuffle=False, sampler=val_sampler,
            num_workers=args.num_workers, collate_fn=vlm_intent_collate_fn,
            pin_memory=True, drop_last=False,
        ) if val_ds is not None else None
    )
    if is_main:
        print(f"world_size={world_size}, per_gpu_bs={per_gpu_bs}, "
              f"train={len(vlm_ds)} val={len(val_ds) if val_ds is not None else 0} "
              f"samples, {len(loader)} train batches/epoch/gpu")

    # Model
    model = VLMIntentDecoder(config).to(device)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"VLMIntentDecoder: {n_params/1e6:.2f}M trainable params")
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
    core = model.module if is_ddp else model

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        core.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        if is_main:
            print(f"Resumed from {args.resume}, starting epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not is_main)
        epoch_loss = 0.0
        train_count = 0
        for batch in pbar:
            vlm_hidden = batch["vlm_hidden"].to(device)
            for k in ["visual_intent_label", "route"]:
                if k in batch:
                    batch[k] = batch[k].to(device)

            pred_intent = model(vlm_hidden)

            loss_dict, log_dict = {}, {}
            core.compute_loss(pred_intent, batch, loss_dict, log_dict)
            loss = loss_dict["loss_visual_intent"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = vlm_hidden.shape[0]
            epoch_loss += loss.item() * n
            train_count += n
            if is_main:
                pbar.set_postfix(loss=loss.item())

        totals = torch.tensor([epoch_loss, train_count], dtype=torch.float64, device=device)
        if is_ddp:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        avg_loss = (totals[0] / totals[1]).item()

        val_loss = None
        if val_loader is not None:
            model.eval()
            val_sum = val_count = 0
            with torch.inference_mode():
                for batch in tqdm(val_loader, desc=f"Val {epoch}", disable=not is_main):
                    vlm_hidden = batch["vlm_hidden"].to(device)
                    for key in ("visual_intent_label", "route"):
                        if key in batch:
                            batch[key] = batch[key].to(device)
                    pred_intent = model(vlm_hidden)
                    loss_dict, log_dict = {}, {}
                    core.compute_loss(pred_intent, batch, loss_dict, log_dict)
                    n = vlm_hidden.shape[0]
                    val_sum += loss_dict["loss_visual_intent"].item() * n
                    val_count += n
            totals = torch.tensor([val_sum, val_count], dtype=torch.float64, device=device)
            if is_ddp:
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            val_loss = (totals[0] / totals[1]).item()

        if is_main:
            print(f"Epoch {epoch}: train={avg_loss:.4f}"
                  + (f" val={val_loss:.4f}" if val_loss is not None else ""))
            improved = val_loss is not None and val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
            if val_loader is not None or (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                ckpt_path = os.path.join(args.logdir, f"model_{epoch:04d}.pth")
                checkpoint = {
                    "epoch": epoch, "model": core.state_dict(),
                    "optimizer": optimizer.state_dict(), "loss": avg_loss,
                    "val_loss": val_loss, "best_val_loss": best_val_loss,
                }
                torch.save(checkpoint, ckpt_path + ".tmp")
                os.replace(ckpt_path + ".tmp", ckpt_path)
                print(f"Saved {ckpt_path}")
                if improved:
                    best_path = os.path.join(args.logdir, "model_best.pth")
                    torch.save(checkpoint, best_path + ".tmp")
                    os.replace(best_path + ".tmp", best_path)
                    print(f"Saved best heldout checkpoint: {best_path}")

    if is_ddp:
        dist.destroy_process_group()
    if is_main:
        print(f"VLM intent training done. Checkpoints in {args.logdir}")


if __name__ == "__main__":
    main()
