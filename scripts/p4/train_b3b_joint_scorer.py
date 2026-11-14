"""Train the frozen-B2, scene-interacting B3b joint trajectory scorer.

Streams matching scene/label shards. Supports torchrun DDP; no LEAD backbone
weights are updated. Held-out routes are used only for model selection/audits.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lead.tfv6.joint_trajectory_scorer import JOINT_HEADS, JointTrajectorySceneScorer, joint_score_loss


def pairs(scene_dir: Path, label_dir: Path, *, max_shards: int = 0):
    scenes = sorted(scene_dir.glob("joint_scene_*.npz"))
    if not scenes:
        raise FileNotFoundError(f"no scene shards in {scene_dir}")
    if max_shards:
        scenes = scenes[:max_shards]
    result = []
    for scene in scenes:
        suffix = scene.stem.removeprefix("joint_scene_")
        label = label_dir / f"joint_labels_{suffix}.npz"
        if not label.exists():
            raise FileNotFoundError(f"missing label shard {label}")
        result.append((scene, label))
    extras = set(label_dir.glob("joint_labels_*.npz")) - {b for _, b in result}
    if extras and not max_shards:
        raise ValueError(f"unmatched label shards: {list(extras)[:3]}")
    return result


def load_pair(scene_path: Path, label_path: Path, *, max_frames: int = 0):
    with np.load(scene_path, allow_pickle=False) as s, np.load(label_path, allow_pickle=False) as l:
        n = min(len(s["keys"]), max_frames) if max_frames else len(s["keys"])
        if len(l["keys"]) < n or not np.array_equal(s["keys"][:n], l["keys"][:n]):
            raise ValueError(f"scene/label key mismatch: {scene_path} {label_path}")
        if str(s["source_checkpoint"]) != str(l["source_checkpoint"]):
            raise ValueError("scene/label checkpoint mismatch")
        if str(scene_path.resolve()) != str(l["source_scene"]):
            raise ValueError("label source scene mismatch")
        scene = s["scene_tokens"][:n].astype(np.float16)
        states = l["candidate_states"][:n].astype(np.float16)
        valid = l["candidate_valid"][:n].astype(bool)
        baseline = l["baseline_index"][:n].astype(np.int64)
        labels = [l[f"label_{name}"][:n].astype(np.float16) for name in JOINT_HEADS]
    if not np.all(valid[np.arange(n), baseline]):
        raise ValueError("baseline candidate invalid")
    # Preserve all valid candidates, but avoid spending compute on empty slots.
    active_count = valid.sum(axis=1)
    max_active = int(active_count.max())
    compact_state = np.zeros((n, max_active, states.shape[2], states.shape[3]), dtype=np.float16)
    compact_valid = np.zeros((n, max_active), dtype=bool)
    compact_baseline = np.zeros(n, dtype=np.int64)
    compact_labels = [np.zeros((n, max_active), dtype=np.float16) for _ in JOINT_HEADS]
    for i in range(n):
        active = np.flatnonzero(valid[i])
        count = len(active)
        compact_state[i, :count] = states[i, active]
        compact_valid[i, :count] = True
        compact_baseline[i] = int(np.flatnonzero(active == baseline[i])[0])
        for j, label in enumerate(labels):
            compact_labels[j][i, :count] = label[i, active]
    return TensorDataset(
        torch.from_numpy(scene), torch.from_numpy(compact_state),
        torch.from_numpy(compact_valid), torch.from_numpy(compact_baseline),
        *(torch.from_numpy(x) for x in compact_labels),
    )


def run_split(model, shard_pairs, device, batch_size, *, optimizer=None, max_frames=0):
    train = optimizer is not None
    model.train(train)
    sums = np.zeros(7, dtype=np.float64)  # loss, examples, raw/selected collision, switches, rescues, introduced
    for scene_path, label_path in shard_pairs:
        dataset = load_pair(scene_path, label_path, max_frames=max_frames)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0,
                            pin_memory=device.type == "cuda")
        for batch in loader:
            scene, states, valid, baseline, *targets = batch
            scene = scene.to(device, dtype=torch.float32, non_blocking=True)
            states = states.to(device, dtype=torch.float32, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            baseline = baseline.to(device, non_blocking=True)
            labels = {name: x.to(device, dtype=torch.float32, non_blocking=True)
                      for name, x in zip(JOINT_HEADS, targets)}
            with torch.set_grad_enabled(train):
                logits = model(states, scene, valid)
                loss, _ = joint_score_loss(logits, labels, valid)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            with torch.no_grad():
                # Diagnostic only: conservative feasibility utility; no runtime
                # policy or threshold is committed by this training script.
                scores = {k: torch.sigmoid(v) for k, v in logits.items()}
                utility = (2 * scores["collision_free"] + scores["ttc"] +
                           scores["drivable"] + scores["task"] +
                           .5 * scores["progress"] + .25 * scores["comfort"] +
                           .5 * scores["imitation"])
                selected = utility.masked_fill(~valid, -1e9).argmax(1)
                rows = torch.arange(len(baseline), device=device)
                raw_unsafe = labels["collision_free"][rows, baseline] < .5
                picked_unsafe = labels["collision_free"][rows, selected] < .5
                sums += [float(loss.item()) * len(rows), len(rows), raw_unsafe.sum().item(),
                         picked_unsafe.sum().item(), (selected != baseline).sum().item(),
                         (raw_unsafe & ~picked_unsafe).sum().item(),
                         (~raw_unsafe & picked_unsafe).sum().item()]
        print(f"processed {scene_path.name}: {len(dataset)} frames", flush=True)
    return sums


def summarize(sums):
    n = max(sums[1], 1)
    return {"loss": sums[0] / n, "frames": int(sums[1]),
            "raw_collision": sums[2] / n, "selected_collision": sums[3] / n,
            "switch": sums[4] / n, "rescue": sums[5] / max(sums[2], 1),
            "introduced": sums[6] / n}


def save_checkpoint(checkpoint, path):
    tmp = path.with_suffix(".tmp.pth")
    torch.save(checkpoint, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-root", type=Path, required=True)
    p.add_argument("--label-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--max-frames-per-shard", type=int, default=0,
                   help="smoke-test only; zero uses all frames")
    p.add_argument("--smoke-shards", type=int, default=0,
                   help="restrict to first N shards per split; testing only")
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--resume", type=Path, help="resume from a matching last checkpoint")
    args = p.parse_args()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train_pairs = pairs(args.scene_root / "train", args.label_root / "train",
                        max_shards=args.smoke_shards)
    val_pairs = pairs(args.scene_root / "heldout", args.label_root / "heldout",
                      max_shards=args.smoke_shards)
    scorer = JointTrajectorySceneScorer(hidden_dim=args.hidden_dim, num_heads=8).to(device)
    model = DDP(scorer, device_ids=[local_rank]) if world > 1 else scorer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    start_epoch = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["source_scene_root"] != str(args.scene_root.resolve()) or saved["source_label_root"] != str(args.label_root.resolve()):
            raise ValueError("resume source directories do not match current run")
        scorer.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"]) + 1
        best = float(saved.get("best_heldout_loss", saved["heldout"]["loss"]))
    for epoch in range(start_epoch, args.epochs):
        random.Random(args.seed + epoch).shuffle(train_pairs)
        local_train = train_pairs[rank::world]
        local_val = val_pairs[rank::world]
        # DDP.join handles unequal last-shard lengths. Metric all-reduction must
        # occur outside the join context; all ranks enter it after their loops.
        context = model.join() if world > 1 else nullcontext()
        with context:
            train_sums = run_split(model, local_train, device, args.batch_size,
                                   optimizer=optimizer,
                                   max_frames=args.max_frames_per_shard)
        with torch.no_grad():
            # Evaluate without DDP forward collectives (ranks may have different
            # counts); the module has identical weights after the join.
            val_sums = run_split(scorer, local_val, device, args.batch_size,
                                 max_frames=args.max_frames_per_shard)
        if world > 1:
            for sums in (train_sums, val_sums):
                reduced = torch.tensor(sums, device=device, dtype=torch.float64)
                dist.all_reduce(reduced)
                sums[:] = reduced.cpu().numpy()
        train_stats, val_stats = summarize(train_sums), summarize(val_sums)
        if rank == 0:
            print(json.dumps({"epoch": epoch, "train": train_stats,
                              "heldout": val_stats}), flush=True)
            improved = val_stats["loss"] < best
            best = min(best, val_stats["loss"])
            ckpt = {"model": scorer.state_dict(), "optimizer": optimizer.state_dict(),
                    "scorer_config": scorer.checkpoint_config(),
                    "heads": JOINT_HEADS, "epoch": epoch, "heldout": val_stats,
                    "best_heldout_loss": best,
                    "validated_for_closed_loop": False,
                    "source_scene_root": str(args.scene_root.resolve()),
                    "source_label_root": str(args.label_root.resolve()),
                    "training_seed": args.seed}
            save_checkpoint(ckpt, args.output / "joint_scorer_last.pth")
            if improved:
                save_checkpoint(ckpt, args.output / "joint_scorer_best.pth")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
