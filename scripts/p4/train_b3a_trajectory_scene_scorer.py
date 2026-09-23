"""Train the B3a-v2 scene-conditioned, multi-head trajectory scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def _validate_metadata(
    cache_dir: str | Path, source_checkpoint: str, source_vocabulary: str
) -> None:
    expected_checkpoint = os.path.normpath(str(Path(source_checkpoint).resolve()))
    expected_vocabulary = os.path.normpath(str(Path(source_vocabulary).resolve()))
    expected_hash = hashlib.sha256(Path(source_vocabulary).read_bytes()).hexdigest()
    paths = sorted(Path(cache_dir).glob("velocity_features_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no feature shards in {cache_dir}")
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            actual_checkpoint = os.path.normpath(str(shard["source_checkpoint"]))
            actual_vocabulary = os.path.normpath(str(shard["source_vocabulary"]))
            actual_hash = str(shard["vocabulary_sha256"])
            required = {
                "scene_tokens",
                "candidate_states",
                "label_collision_free",
                "label_ttc",
                "label_progress",
                "label_comfort",
                "label_imitation",
            }
            missing = required - set(shard.files)
            scene_v2 = bool(shard.get("scene_v2", np.asarray(False)))
            full_reachability = bool(
                shard.get("full_profile_reachability", np.asarray(False))
            )
        if missing:
            raise ValueError(
                f"{path} is an old B3a-v1 cache; missing {sorted(missing)}"
            )
        if not scene_v2 or not full_reachability:
            raise ValueError(
                f"{path} was not built with B3a-v2 scene/full-reachability flags"
            )
        if actual_checkpoint != expected_checkpoint:
            raise ValueError(f"checkpoint mismatch in {path}")
        if actual_vocabulary != expected_vocabulary or actual_hash != expected_hash:
            raise ValueError(f"velocity vocabulary mismatch in {path}")


def _dataset(arrays) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(arrays.scene_tokens),
        torch.from_numpy(arrays.candidate_states),
        torch.from_numpy(arrays.candidate_valid),
        torch.from_numpy(arrays.label_collision_free),
        torch.from_numpy(arrays.label_ttc),
        torch.from_numpy(arrays.label_progress),
        torch.from_numpy(arrays.label_comfort),
        torch.from_numpy(arrays.label_imitation),
        torch.from_numpy(arrays.collision),
        torch.from_numpy(arrays.imitation_error),
    )


def _to_device(batch, device: torch.device):
    (
        scene,
        state,
        valid,
        collision_free,
        ttc,
        progress,
        comfort,
        imitation,
        collision,
        error,
    ) = batch
    labels = {
        "collision_free": collision_free.to(
            device, dtype=torch.float32, non_blocking=True
        ),
        "ttc": ttc.to(device, dtype=torch.float32, non_blocking=True),
        "progress": progress.to(device, dtype=torch.float32, non_blocking=True),
        "comfort": comfort.to(device, dtype=torch.float32, non_blocking=True),
        "imitation": imitation.to(device, dtype=torch.float32, non_blocking=True),
    }
    return (
        scene.to(device, dtype=torch.float32, non_blocking=True),
        state.to(device, dtype=torch.float32, non_blocking=True),
        valid.to(device, dtype=torch.bool, non_blocking=True),
        labels,
        collision.to(device, dtype=torch.bool, non_blocking=True),
        error.to(device, dtype=torch.float32, non_blocking=True),
    )


def _unwrap_scorer(scorer):
    if isinstance(scorer, torch.nn.DataParallel):
        return scorer.module
    return scorer


def _safety_gate_failures(metrics: dict, args: argparse.Namespace) -> list[str]:
    """Require measured selection to improve safety with limited interventions."""

    checks = (
        (
            metrics["selected_collision_rate"] <= metrics["raw_collision_rate"],
            "selected collisions exceed raw",
        ),
        (
            metrics["introduced_collision_rate"] <= metrics["rescue_rate"],
            "introduced collisions exceed rescues",
        ),
        (metrics["switch_rate"] <= args.max_switch_rate, "switch rate too high"),
        (
            metrics["fallback_rate"] <= args.max_fallback_rate,
            "fallback rate too high",
        ),
        (
            metrics["collision_auroc"] >= args.min_collision_auroc,
            "candidate collision AUROC too low",
        ),
        (
            metrics["within_frame_safe_above_unsafe"]
            >= args.min_within_frame_safety_rank,
            "within-frame safety ranking too weak",
        ),
        (
            metrics["selected_profile_mae_mps"]
            <= metrics["raw_profile_mae_mps"] + args.max_mae_increase,
            "selected profile MAE increase too high",
        ),
    )
    return [reason for passed, reason in checks if not passed]


@torch.inference_mode()
def _evaluate(
    scorer,
    loader,
    device,
    trajectory_score_loss,
    select_scene_scored_trajectory,
    binary_auroc,
    binary_average_precision,
    collision_unsafe_weight,
    collision_pair_weight,
    ttc_pair_weight,
) -> dict[str, float | int]:
    scorer.eval()
    scorer_module = _unwrap_scorer(scorer)
    total_loss = 0.0
    frames = valid_count = collisions = selected_collisions = fallback = 0
    raw_collisions = rescue = introduced = switches = 0
    ranking_correct = ranking_tied = ranking_pairs = 0
    selected_error = selected_progress = raw_error = 0.0
    head_loss_sum = {name: 0.0 for name in scorer_module.heads}
    risk_scores, risk_labels = [], []
    for batch in tqdm(loader, desc="held-out", unit="batch", leave=False):
        scene, state, valid, labels, collision, error = _to_device(batch, device)
        logits = scorer(state, scene, valid)
        loss, head_losses = trajectory_score_loss(
            logits,
            labels,
            valid,
            collision_unsafe_weight=collision_unsafe_weight,
            collision_pair_weight=collision_pair_weight,
            ttc_pair_weight=ttc_pair_weight,
        )
        selection = select_scene_scored_trajectory(logits, valid)
        count = len(scene)
        rows = torch.arange(count, device=device)
        total_loss += float(loss) * count
        frames += count
        valid_count += int(valid.sum())
        collisions += int(collision[valid].sum())
        selected_collisions += int(collision[rows, selection.selected_index].sum())
        raw = collision[:, 0]
        selected = collision[rows, selection.selected_index]
        raw_collisions += int(raw.sum())
        rescue += int((raw & ~selected).sum())
        introduced += int((~raw & selected).sum())
        switches += int((selection.selected_index != 0).sum())
        raw_error += float(error[:, 0].sum())
        safe = valid & ~collision
        unsafe = valid & collision
        mixed = safe.any(1) & unsafe.any(1)
        if mixed.any():
            score = selection.scores["collision_free"][mixed]
            comparison = score[:, :, None] - score[:, None, :]
            pairs = safe[mixed][:, :, None] & unsafe[mixed][:, None, :]
            ranking_correct += int(((comparison > 0) & pairs).sum())
            ranking_tied += int(((comparison == 0) & pairs).sum())
            ranking_pairs += int(pairs.sum())
        fallback += int(selection.fallback.sum())
        selected_error += float(error[rows, selection.selected_index].sum())
        selected_progress += float(
            labels["progress"][rows, selection.selected_index].sum()
        )
        for name in head_loss_sum:
            head_loss_sum[name] += float(head_losses[name]) * count
        risk_scores.append(
            (1.0 - selection.scores["collision_free"][valid]).cpu().numpy()
        )
        risk_labels.append(collision[valid].cpu().numpy())
    risk_score = np.concatenate(risk_scores)
    risk_label = np.concatenate(risk_labels)
    denom = max(frames, 1)
    return {
        "loss": total_loss / denom,
        **{f"{name}_loss": value / denom for name, value in head_loss_sum.items()},
        "collision_auroc": binary_auroc(risk_score, risk_label),
        "collision_average_precision": binary_average_precision(risk_score, risk_label),
        "candidate_collision_rate": collisions / max(valid_count, 1),
        "raw_collision_rate": raw_collisions / denom,
        "selected_collision_rate": selected_collisions / denom,
        "rescue_rate": rescue / denom,
        "introduced_collision_rate": introduced / denom,
        "switch_rate": switches / denom,
        "within_frame_safe_above_unsafe": (ranking_correct + 0.5 * ranking_tied)
        / max(ranking_pairs, 1),
        "within_frame_safety_pairs": ranking_pairs,
        "raw_profile_mae_mps": raw_error / denom,
        "selected_profile_mae_mps": selected_error / denom,
        "selected_progress_score": selected_progress / denom,
        "fallback_rate": fallback / denom,
        "n_frames": frames,
        "n_valid_candidates": valid_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache-dir", required=True)
    parser.add_argument("--val-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-vocabulary", required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--interaction-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="global held-out batch size; defaults to the training batch size",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-unsafe-weight", type=float, default=20.0)
    parser.add_argument("--collision-pair-weight", type=float, default=2.0)
    parser.add_argument("--ttc-pair-weight", type=float, default=1.0)
    parser.add_argument("--init-scorer", default=None)
    parser.add_argument("--max-switch-rate", type=float, default=0.10)
    parser.add_argument("--max-fallback-rate", type=float, default=0.05)
    parser.add_argument("--min-collision-auroc", type=float, default=0.80)
    parser.add_argument("--min-within-frame-safety-rank", type=float, default=0.60)
    parser.add_argument("--max-mae-increase", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="use all visible CUDA devices with one shared host-side feature cache",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from lead.tfv6.route_safety_head import binary_auroc, binary_average_precision
    from lead.tfv6.trajectory_scene_scorer import (
        TrajectorySceneScorer,
        load_trajectory_scene_feature_dir,
        select_scene_scored_trajectory,
        trajectory_score_loss,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    visible_gpus = torch.cuda.device_count()
    if args.data_parallel and visible_gpus < 2:
        raise RuntimeError(
            f"--data-parallel requires at least 2 visible GPUs, found {visible_gpus}"
        )
    _validate_metadata(
        args.train_cache_dir, args.source_checkpoint, args.source_vocabulary
    )
    _validate_metadata(
        args.val_cache_dir, args.source_checkpoint, args.source_vocabulary
    )
    print(f"loading B3a-v2 train features from {args.train_cache_dir}")
    train_arrays = load_trajectory_scene_feature_dir(args.train_cache_dir)
    print(f"loading B3a-v2 held-out features from {args.val_cache_dir}")
    val_arrays = load_trajectory_scene_feature_dir(args.val_cache_dir)
    if set(train_arrays.keys.tolist()) & set(val_arrays.keys.tolist()):
        raise ValueError("train and held-out caches overlap")
    train_dataset = _dataset(train_arrays)
    val_dataset = _dataset(val_arrays)
    train_collision_rate = float(
        train_arrays.collision[train_arrays.candidate_valid].mean()
    )
    unsafe = int(train_arrays.collision[train_arrays.candidate_valid].sum())
    safe = int(train_arrays.candidate_valid.sum()) - unsafe
    if unsafe == 0:
        raise ValueError("training cache contains no unsafe counterfactual candidates")
    collision_unsafe_weight = min(safe / unsafe, args.max_unsafe_weight)
    del train_arrays, val_arrays

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
        batch_size=args.val_batch_size or args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    scene_dim = int(train_dataset.tensors[0].shape[-1])
    profile_steps = int(train_dataset.tensors[1].shape[-2])
    state_dim = int(train_dataset.tensors[1].shape[-1])
    scorer_module = TrajectorySceneScorer(
        state_dim=state_dim,
        scene_dim=scene_dim,
        hidden_dim=args.hidden_dim,
        profile_steps=profile_steps,
        num_heads=args.num_heads,
        temporal_layers=args.temporal_layers,
        interaction_layers=args.interaction_layers,
        dropout=args.dropout,
    ).to(device)
    if args.init_scorer is not None:
        initial = torch.load(args.init_scorer, map_location="cpu", weights_only=True)
        if initial["scorer_config"] != scorer_module.checkpoint_config():
            raise ValueError("initial scorer architecture differs from this run")
        for field, expected in (
            ("source_checkpoint", args.source_checkpoint),
            ("source_vocabulary", args.source_vocabulary),
        ):
            if os.path.normpath(
                str(Path(initial[field]).resolve())
            ) != os.path.normpath(str(Path(expected).resolve())):
                raise ValueError(f"initial scorer {field} differs from this run")
        scorer_module.load_state_dict(initial["model"])
        print(f"initialized scorer weights from {args.init_scorer}")
    if args.data_parallel:
        scorer = torch.nn.DataParallel(
            scorer_module, device_ids=list(range(visible_gpus))
        )
        parallel_mode = "data_parallel"
        print(
            f"training with DataParallel on {visible_gpus} GPUs; "
            f"global batch={args.batch_size} "
            f"(~{args.batch_size / visible_gpus:.1f}/GPU)"
        )
    else:
        scorer = scorer_module
        parallel_mode = "single_device"
        print(f"training on {device}; global batch={args.batch_size}")
    optimizer = torch.optim.AdamW(
        scorer.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_checkpoint_path = output_dir / f"trajectory_scene_scorer_safe_{run_id}.pth"
    config = vars(args) | {
        "run_id": run_id,
        "scene_dim": scene_dim,
        "scene_tokens": int(train_dataset.tensors[0].shape[1]),
        "state_dim": state_dim,
        "profile_steps": profile_steps,
        "candidate_count": int(train_dataset.tensors[1].shape[1]),
        "train_frames": len(train_dataset),
        "val_frames": len(val_dataset),
        "train_collision_rate": train_collision_rate,
        "collision_unsafe_weight": collision_unsafe_weight,
        "device": str(device),
        "parallel_mode": parallel_mode,
        "visible_gpus": visible_gpus,
    }
    with (output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)

    history = []
    best_loss = float("inf")
    best_risk_auroc = float("-inf")
    best_safe_key = None
    best_safe_epoch = None
    for epoch in range(args.epochs):
        scorer.train()
        train_loss = frame_count = 0.0
        for batch in tqdm(
            train_loader,
            desc=f"train {epoch + 1:02d}/{args.epochs:02d}",
            unit="batch",
        ):
            scene, state, valid, labels, _, _ = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = scorer(state, scene, valid)
                loss, _ = trajectory_score_loss(
                    logits,
                    labels,
                    valid,
                    collision_unsafe_weight=collision_unsafe_weight,
                    collision_pair_weight=args.collision_pair_weight,
                    ttc_pair_weight=args.ttc_pair_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(scene)
            train_loss += float(loss.detach()) * count
            frame_count += count
        scheduler.step()
        metrics = _evaluate(
            scorer,
            val_loader,
            device,
            trajectory_score_loss,
            select_scene_scored_trajectory,
            binary_auroc,
            binary_average_precision,
            collision_unsafe_weight,
            args.collision_pair_weight,
            args.ttc_pair_weight,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(frame_count, 1),
            "lr": optimizer.param_groups[0]["lr"],
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        gate_failures = _safety_gate_failures(metrics, args)
        row["safety_gate_failures"] = gate_failures
        history.append(row)
        print(
            f"epoch={epoch:02d} train={row['train_loss']:.4f} "
            f"val={metrics['loss']:.4f} risk_AUROC={metrics['collision_auroc']:.4f} "
            f"selected_collision={metrics['selected_collision_rate']:.2%} "
            f"raw_collision={metrics['raw_collision_rate']:.2%} "
            f"switch={metrics['switch_rate']:.1%} "
            f"within_frame_rank={metrics['within_frame_safe_above_unsafe']:.3f} "
            f"MAE={metrics['selected_profile_mae_mps']:.3f} "
            f"progress={metrics['selected_progress_score']:.3f} "
            f"safety_gate={'pass' if not gate_failures else 'fail'}"
        )
        checkpoint = {
            # Always save the underlying scorer without a ``module.`` prefix so
            # the same checkpoint can be consumed by single-GPU closed loop.
            "model": scorer_module.state_dict(),
            "scorer_config": scorer_module.checkpoint_config(),
            "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
            "source_vocabulary": str(Path(args.source_vocabulary).resolve()),
            "epoch": epoch,
            "metrics": metrics,
            "score_heads": list(scorer_module.heads),
            "collision_unsafe_weight": collision_unsafe_weight,
            "collision_pair_weight": args.collision_pair_weight,
            "ttc_pair_weight": args.ttc_pair_weight,
            "run_id": run_id,
            "safety_qualified": not gate_failures,
            "safety_gate_failures": gate_failures,
        }
        torch.save(checkpoint, output_dir / "trajectory_scene_scorer_last.pth")
        if metrics["loss"] < best_loss:
            best_loss = float(metrics["loss"])
            torch.save(checkpoint, output_dir / "trajectory_scene_scorer_best.pth")
        if metrics["collision_auroc"] > best_risk_auroc:
            best_risk_auroc = float(metrics["collision_auroc"])
            torch.save(
                checkpoint, output_dir / "trajectory_scene_scorer_best_risk_auroc.pth"
            )
        if not gate_failures:
            safe_key = (
                metrics["selected_collision_rate"],
                metrics["introduced_collision_rate"],
                metrics["selected_profile_mae_mps"],
                -metrics["within_frame_safe_above_unsafe"],
            )
            if best_safe_key is None or safe_key < best_safe_key:
                best_safe_key = safe_key
                best_safe_epoch = epoch
                torch.save(checkpoint, safe_checkpoint_path)
    with (output_dir / "history.json").open("w") as handle:
        json.dump(history, handle, indent=2)
    selection_status = {
        "run_id": run_id,
        "best_loss_epoch": min(history, key=lambda row: row["val_loss"])["epoch"],
        "best_risk_auroc_epoch": max(
            history, key=lambda row: row["val_collision_auroc"]
        )["epoch"],
        "best_safe_epoch": best_safe_epoch,
        "safe_checkpoint": str(safe_checkpoint_path)
        if best_safe_epoch is not None
        else None,
        "gate": {
            "max_switch_rate": args.max_switch_rate,
            "max_fallback_rate": args.max_fallback_rate,
            "min_collision_auroc": args.min_collision_auroc,
            "min_within_frame_safety_rank": args.min_within_frame_safety_rank,
            "max_mae_increase_mps": args.max_mae_increase,
            "selected_collision_no_worse_than_raw": True,
            "introduced_no_more_than_rescued": True,
        },
    }
    with (output_dir / "selection_status.json").open("w") as handle:
        json.dump(selection_status, handle, indent=2)
    print(
        f"best held-out loss={best_loss:.4f}; wrote "
        f"{output_dir / 'trajectory_scene_scorer_best.pth'}"
    )
    print(
        f"best safety-qualified epoch={best_safe_epoch}; "
        f"checkpoint={selection_status['safe_checkpoint']}"
    )


if __name__ == "__main__":
    main()
