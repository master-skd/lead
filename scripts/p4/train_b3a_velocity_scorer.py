"""Train the B3a velocity scorer on frozen winner-route feature caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _validate_cache_metadata(
    cache_dir: str | Path, source_checkpoint: str, source_vocabulary: str
) -> None:
    expected_checkpoint = os.path.normpath(str(Path(source_checkpoint).resolve()))
    expected_vocabulary = os.path.normpath(str(Path(source_vocabulary).resolve()))
    expected_hash = hashlib.sha256(Path(source_vocabulary).read_bytes()).hexdigest()
    shards = sorted(Path(cache_dir).glob("velocity_features_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no velocity feature shards in {cache_dir}")
    for path in shards:
        with np.load(path, allow_pickle=False) as shard:
            actual_checkpoint = os.path.normpath(str(shard["source_checkpoint"]))
            actual_vocabulary = os.path.normpath(str(shard["source_vocabulary"]))
            actual_hash = str(shard["vocabulary_sha256"])
        if actual_checkpoint != expected_checkpoint:
            raise ValueError(f"checkpoint mismatch in {path}")
        if actual_vocabulary != expected_vocabulary or actual_hash != expected_hash:
            raise ValueError(f"velocity vocabulary mismatch in {path}")


def _dataset(arrays) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(arrays.route_features),
        torch.from_numpy(arrays.current_speed),
        torch.from_numpy(arrays.raw_target_speed),
        torch.from_numpy(arrays.candidate_velocity),
        torch.from_numpy(arrays.candidate_valid),
        torch.from_numpy(arrays.collision),
        torch.from_numpy(arrays.imitation_target.astype(np.int64)),
        torch.from_numpy(arrays.imitation_error),
    )


def _to_device(batch, device: torch.device):
    route, current, raw, velocity, valid, collision, target, error = batch
    return (
        route.to(device, dtype=torch.float32, non_blocking=True),
        current.to(device, dtype=torch.float32, non_blocking=True),
        raw.to(device, dtype=torch.float32, non_blocking=True),
        velocity.to(device, dtype=torch.float32, non_blocking=True),
        valid.to(device, dtype=torch.bool, non_blocking=True),
        collision.to(device, dtype=torch.float32, non_blocking=True),
        target.to(device, dtype=torch.long, non_blocking=True),
        error.to(device, dtype=torch.float32, non_blocking=True),
    )


@torch.inference_mode()
def _evaluate(
    scorer,
    loader,
    device,
    pos_weight,
    collision_loss_weight,
    binary_auroc,
    binary_average_precision,
) -> dict:
    scorer.eval()
    totals = {
        "loss": 0.0,
        "imitation_loss": 0.0,
        "collision_loss": 0.0,
        "frames": 0,
        "valid_candidates": 0,
        "correct": 0,
        "selected_error": 0.0,
        "oracle_error": 0.0,
    }
    risk_scores, risk_labels = [], []
    for batch in loader:
        route, current, raw, velocity, valid, collision, target, error = _to_device(
            batch, device
        )
        preference, risk = scorer(route, current, raw, velocity)
        masked_preference = preference.masked_fill(~valid, float("-inf"))
        imitation_loss = F.cross_entropy(masked_preference, target)
        element_loss = F.binary_cross_entropy_with_logits(
            risk, collision, pos_weight=pos_weight, reduction="none"
        )
        collision_loss = (element_loss * valid).sum() / valid.sum().clamp(min=1)
        loss = imitation_loss + collision_loss_weight * collision_loss
        count = len(route)
        selected = masked_preference.argmax(dim=1)
        rows = torch.arange(count, device=device)
        totals["loss"] += float(loss) * count
        totals["imitation_loss"] += float(imitation_loss) * count
        totals["collision_loss"] += float(collision_loss) * count
        totals["frames"] += count
        totals["valid_candidates"] += int(valid.sum())
        totals["correct"] += int((selected == target).sum())
        totals["selected_error"] += float(error[rows, selected].sum())
        totals["oracle_error"] += float(error[rows, target].sum())
        risk_scores.append(torch.sigmoid(risk[valid]).cpu().numpy())
        risk_labels.append(collision[valid].bool().cpu().numpy())
    scores = np.concatenate(risk_scores)
    labels = np.concatenate(risk_labels)
    frames = max(int(totals["frames"]), 1)
    return {
        "loss": totals["loss"] / frames,
        "imitation_loss": totals["imitation_loss"] / frames,
        "collision_loss": totals["collision_loss"] / frames,
        "imitation_top1_accuracy": totals["correct"] / frames,
        "selected_profile_mae_mps": totals["selected_error"] / frames,
        "oracle_profile_mae_mps": totals["oracle_error"] / frames,
        "collision_auroc": binary_auroc(scores, labels),
        "collision_average_precision": binary_average_precision(scores, labels),
        "collision_rate": float(labels.mean()),
        "n_frames": int(totals["frames"]),
        "n_valid_candidates": int(totals["valid_candidates"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache-dir", required=True)
    parser.add_argument("--val-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-vocabulary", required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--collision-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-pos-weight", type=float, default=30.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from lead.tfv6.route_safety_head import binary_auroc, binary_average_precision
    from lead.tfv6.velocity_scorer import VelocityScorer, load_velocity_feature_dir

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _validate_cache_metadata(
        args.train_cache_dir, args.source_checkpoint, args.source_vocabulary
    )
    _validate_cache_metadata(
        args.val_cache_dir, args.source_checkpoint, args.source_vocabulary
    )
    print(f"loading train velocity features from {args.train_cache_dir}")
    train_arrays = load_velocity_feature_dir(args.train_cache_dir)
    print(f"loading held-out velocity features from {args.val_cache_dir}")
    val_arrays = load_velocity_feature_dir(args.val_cache_dir)
    if set(train_arrays.keys.tolist()) & set(val_arrays.keys.tolist()):
        raise ValueError("train and held-out velocity caches overlap")
    train_dataset = _dataset(train_arrays)
    val_dataset = _dataset(val_arrays)
    train_labels = train_arrays.collision[train_arrays.candidate_valid]
    positives = int(train_labels.sum())
    negatives = len(train_labels) - positives
    if positives == 0:
        raise ValueError("training cache contains no positive collision labels")
    pos_weight_value = min(negatives / positives, args.max_pos_weight)
    del train_arrays, val_arrays, train_labels

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    route_dim = int(train_dataset.tensors[0].shape[-1])
    profile_steps = int(train_dataset.tensors[3].shape[-1])
    scorer = VelocityScorer(
        route_feature_dim=route_dim,
        profile_steps=profile_steps,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        scorer.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    pos_weight = torch.tensor(pos_weight_value, device=device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "route_feature_dim": route_dim,
        "profile_steps": profile_steps,
        "candidate_count": int(train_dataset.tensors[3].shape[1]),
        "train_frames": len(train_dataset),
        "val_frames": len(val_dataset),
        "train_valid_candidates": positives + negatives,
        "train_collision_rate": positives / max(positives + negatives, 1),
        "pos_weight": pos_weight_value,
        "device": str(device),
    }
    with (output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)

    history = []
    best_loss = float("inf")
    for epoch in range(args.epochs):
        scorer.train()
        loss_sum = imitation_sum = collision_sum = 0.0
        frame_count = 0
        for batch in train_loader:
            route, current, raw, velocity, valid, collision, target, _ = _to_device(
                batch, device
            )
            optimizer.zero_grad(set_to_none=True)
            preference, risk = scorer(route, current, raw, velocity)
            imitation_loss = F.cross_entropy(
                preference.masked_fill(~valid, float("-inf")), target
            )
            element_loss = F.binary_cross_entropy_with_logits(
                risk, collision, pos_weight=pos_weight, reduction="none"
            )
            collision_loss = (element_loss * valid).sum() / valid.sum().clamp(min=1)
            loss = imitation_loss + args.collision_loss_weight * collision_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.0)
            optimizer.step()
            count = len(route)
            loss_sum += float(loss.detach()) * count
            imitation_sum += float(imitation_loss.detach()) * count
            collision_sum += float(collision_loss.detach()) * count
            frame_count += count
        scheduler.step()
        metrics = _evaluate(
            scorer,
            val_loader,
            device,
            pos_weight,
            args.collision_loss_weight,
            binary_auroc,
            binary_average_precision,
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(frame_count, 1),
            "train_imitation_loss": imitation_sum / max(frame_count, 1),
            "train_collision_loss": collision_sum / max(frame_count, 1),
            "lr": optimizer.param_groups[0]["lr"],
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train={row['train_loss']:.4f} "
            f"val={metrics['loss']:.4f} top1={metrics['imitation_top1_accuracy']:.2%} "
            f"MAE={metrics['selected_profile_mae_mps']:.3f} "
            f"risk AUROC={metrics['collision_auroc']:.4f} "
            f"AP={metrics['collision_average_precision']:.4f}"
        )
        checkpoint = {
            "model": scorer.state_dict(),
            "scorer_config": scorer.checkpoint_config(),
            "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
            "source_vocabulary": str(Path(args.source_vocabulary).resolve()),
            "epoch": epoch,
            "metrics": metrics,
            "pos_weight": pos_weight_value,
        }
        torch.save(checkpoint, output_dir / "velocity_scorer_last.pth")
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(checkpoint, output_dir / "velocity_scorer_best.pth")
    with (output_dir / "history.json").open("w") as handle:
        json.dump(history, handle, indent=2)
    print(
        f"best held-out loss={best_loss:.4f}; wrote "
        f"{output_dir / 'velocity_scorer_best.pth'}"
    )


if __name__ == "__main__":
    main()
