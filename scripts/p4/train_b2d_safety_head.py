"""Train the B2d dynamic-risk head on frozen corridor arm features."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _validate_source_checkpoint(cache_dir: str, expected: str) -> None:
    expected = os.path.normpath(expected)
    shard_paths = sorted(Path(cache_dir).glob("safety_features_*.npz"))
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as shard:
            if "source_checkpoint" not in shard:
                raise ValueError(f"{path} has no source_checkpoint metadata")
            actual = os.path.normpath(str(shard["source_checkpoint"]))
        if actual != expected:
            raise ValueError(f"feature/checkpoint mismatch in {path}: {actual} != {expected}")


def _valid_arm_dataset(arrays) -> TensorDataset:
    frame, arm = np.nonzero(arrays.valid)
    return TensorDataset(
        torch.from_numpy(arrays.route_features[frame, arm].astype(np.float32)),
        torch.from_numpy(arrays.current_speed[frame].astype(np.float32)),
        torch.from_numpy(arrays.target_speed[frame].astype(np.float32)),
        torch.from_numpy(arrays.collision[frame, arm].astype(np.float32)),
    )


@torch.inference_mode()
def _evaluate(head, loader, device, binary_auroc, binary_average_precision) -> dict:
    head.eval()
    scores, labels, loss_sum, count = [], [], 0.0, 0
    for features, current_speed, target_speed, label in loader:
        features = features.to(device, non_blocking=True)
        current_speed = current_speed.to(device, non_blocking=True)
        target_speed = target_speed.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        logits = head(features, current_speed, target_speed)
        loss_sum += float(F.binary_cross_entropy_with_logits(logits, label, reduction="sum"))
        count += len(label)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(label.cpu().numpy())
    score = np.concatenate(scores)
    label = np.concatenate(labels).astype(bool)
    return {
        "loss_unweighted": loss_sum / max(count, 1),
        "auroc": binary_auroc(score, label),
        "average_precision": binary_average_precision(score, label),
        "collision_rate": float(label.mean()),
        "n_valid_arms": int(count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache-dir", required=True)
    parser.add_argument("--val-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-pos-weight", type=float, default=30.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from lead.tfv6.route_safety_head import (
        RouteSafetyHead,
        binary_average_precision,
        binary_auroc,
        load_safety_feature_dir,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"loading train features from {args.train_cache_dir}")
    _validate_source_checkpoint(args.train_cache_dir, args.source_checkpoint)
    _validate_source_checkpoint(args.val_cache_dir, args.source_checkpoint)
    train_arrays = load_safety_feature_dir(args.train_cache_dir)
    print(f"loading held-out features from {args.val_cache_dir}")
    val_arrays = load_safety_feature_dir(args.val_cache_dir)
    if set(train_arrays.keys.tolist()) & set(val_arrays.keys.tolist()):
        raise ValueError("train and held-out feature caches overlap")
    train_dataset = _valid_arm_dataset(train_arrays)
    val_dataset = _valid_arm_dataset(val_arrays)
    del train_arrays, val_arrays

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    route_feature_dim = int(train_dataset.tensors[0].shape[-1])
    head = RouteSafetyHead(
        route_feature_dim=route_feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    labels = train_dataset.tensors[-1]
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0:
        raise ValueError("training cache contains no colliding valid arm")
    pos_weight_value = min(negatives / positives, args.max_pos_weight)
    pos_weight = torch.tensor(pos_weight_value, device=device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "route_feature_dim": route_feature_dim,
        "train_valid_arms": len(train_dataset),
        "val_valid_arms": len(val_dataset),
        "train_collision_rate": positives / len(labels),
        "pos_weight": pos_weight_value,
        "device": str(device),
    }
    with (output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)

    history = []
    best_ap = -float("inf")
    for epoch in range(args.epochs):
        head.train()
        weighted_sum = 0.0
        count = 0
        for features, current_speed, target_speed, label in train_loader:
            features = features.to(device, non_blocking=True)
            current_speed = current_speed.to(device, non_blocking=True)
            target_speed = target_speed.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features, current_speed, target_speed)
            loss = F.binary_cross_entropy_with_logits(logits, label, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            weighted_sum += float(loss.detach()) * len(label)
            count += len(label)
        scheduler.step()
        metrics = _evaluate(head, val_loader, device, binary_auroc, binary_average_precision)
        row = {
            "epoch": epoch,
            "train_weighted_loss": weighted_sum / max(count, 1),
            "lr": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train={row['train_weighted_loss']:.4f} "
            f"val={row['loss_unweighted']:.4f} AUROC={row['auroc']:.4f} "
            f"AP={row['average_precision']:.4f}"
        )
        checkpoint = {
            "model": head.state_dict(),
            "head_config": head.checkpoint_config(),
            "source_checkpoint": args.source_checkpoint,
            "epoch": epoch,
            "metrics": metrics,
            "pos_weight": pos_weight_value,
        }
        torch.save(checkpoint, output_dir / "safety_head_last.pth")
        if metrics["average_precision"] > best_ap:
            best_ap = metrics["average_precision"]
            torch.save(checkpoint, output_dir / "safety_head_best.pth")
    with (output_dir / "history.json").open("w") as handle:
        json.dump(history, handle, indent=2)
    print(f"best held-out AP={best_ap:.4f}; wrote {output_dir / 'safety_head_best.pth'}")


if __name__ == "__main__":
    main()
