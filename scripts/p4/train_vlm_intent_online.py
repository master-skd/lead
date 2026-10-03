"""Train the P6 intent decoder from RGB, without a persistent VLM feature cache.

Each DDP rank owns one frozen Qwen model. Qwen is run in small inference-only
microbatches; its image-token activations stay on the GPU and are concatenated
before the decoder step, preserving the original decoder batch size.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from lead.inference.vlm_feature_extractor import PROMPT_3CAM_DRIVABLE
from lead.tfv6.vlm_intent_decoder import VLMIntentDecoder


class OnlineIntentDataset(Dataset):
    """Only RGB and precomputed lane-graph labels; no CARLA sensor/bucket cache."""

    def __init__(self, manifest: Path, label_dir: Path, limit: int = 0):
        self.samples = []
        with manifest.open() as stream:
            for line in stream:
                entry = json.loads(line)
                label = label_dir / entry["scenario"] / entry["route"] / (entry["frame"] + ".npy")
                self.samples.append((Path(entry["src"]), label))
                if limit and len(self.samples) >= limit:
                    break
        if not self.samples:
            raise ValueError(f"empty intent manifest: {manifest}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, torch.Tensor]:
        import numpy as np

        image_path, label_path = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")  # full three-camera strip, no crop
        label = torch.from_numpy(np.load(label_path).astype("float32"))
        if image.size != (1152, 384) or label.shape != (1, 320, 384):
            raise ValueError(f"invalid image or label shape: {image_path}, {label_path}")
        return image, label


def collate_online(batch: list[tuple[Image.Image, torch.Tensor]]):
    return [image for image, _ in batch], torch.stack([label for _, label in batch])


def qwen_image_tokens(images, model, processor, image_token_id: int, merge: int, microbatch: int, device):
    """Same prompt, processor and fp16 boundary as extract_vlm_p6_3cam.py."""
    pieces = []
    for offset in range(0, len(images), microbatch):
        chunk = images[offset : offset + microbatch]
        conversations = [
            [{"role": "user", "content": [
                {"type": "text", "text": PROMPT_3CAM_DRIVABLE},
                {"type": "image", "image": image},
            ]}]
            for image in chunk
        ]
        inputs = processor.apply_chat_template(
            conversations[0] if len(chunk) == 1 else conversations,
            tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            processor_kwargs={"padding": len(chunk) > 1},
        ).to(device)
        with torch.inference_mode():
            result = model(**inputs, output_hidden_states=True, use_cache=False)
            hidden = result.hidden_states[-1]
            if inputs["image_grid_thw"].shape[0] != len(chunk):
                raise ValueError("Qwen returned the wrong number of image grids")
            for slot, grid in enumerate(inputs["image_grid_thw"].tolist()):
                height, width = grid[1] // merge, grid[2] // merge
                tokens = hidden[slot][inputs["input_ids"][slot] == image_token_id]
                if (height, width, tokens.shape[0], tokens.shape[-1]) != (12, 36, 432, 2560):
                    raise ValueError(f"unexpected Qwen image-token shape: {height}x{width}, {tuple(tokens.shape)}")
                # Offline extraction stores fp16 .npy; reproduce that quantization.
                pieces.append(tokens.reshape(12, 36, 2560).to(torch.float16).clone())
        del result, inputs, hidden
    return torch.stack(pieces).float()


def reduce_average(total: float, count: int, device) -> float:
    sums = torch.tensor([total, count], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    return (sums[0] / sums[1]).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--split-metadata", type=Path, required=True)
    parser.add_argument("--lanegraph-label-dir", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256, help="global decoder batch")
    parser.add_argument("--qwen-batch-size", type=int, default=4, help="Qwen forward microbatch")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if args.batch_size < world_size or args.batch_size % world_size or args.qwen_batch_size < 1:
        parser.error("batch-size must be divisible by world size; qwen-batch-size must be positive")
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    train_ds = OnlineIntentDataset(args.train_manifest, args.lanegraph_label_dir, args.limit_train)
    val_ds = OnlineIntentDataset(args.val_manifest, args.lanegraph_label_dir, args.limit_val)
    split = json.loads(args.split_metadata.read_text())
    if split["route_overlap"] or (
        not args.limit_train and len(train_ds) != split["train_frames"]
    ) or (
        not args.limit_val and len(val_ds) != split["heldout_frames"]
    ):
        raise ValueError("online intent train/heldout split does not match metadata")
    per_gpu_batch = args.batch_size // world_size
    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None
    loader_kwargs = dict(
        batch_size=per_gpu_batch, num_workers=args.num_workers,
        collate_fn=collate_online, pin_memory=True,
    )
    train_loader = DataLoader(
        train_ds, sampler=train_sampler, shuffle=train_sampler is None,
        drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, sampler=val_sampler, shuffle=False, **loader_kwargs)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.qwen_model, trust_remote_code=True)
    qwen = AutoModelForImageTextToText.from_pretrained(
        args.qwen_model, dtype=torch.bfloat16, device_map=f"cuda:{local_rank}",
        trust_remote_code=True,
    ).eval()
    qwen.requires_grad_(False)
    image_token_id = qwen.config.image_token_id
    merge = getattr(qwen.config.vision_config, "spatial_merge_size", 2)

    # These are the TrainingConfig values used by the cached-feature recipe.
    config = SimpleNamespace(
        lidar_height_pixel=320, lidar_width_pixel=384,
        visual_intent_loss_weight=1.0, intent_tversky_weight=0.75,
        intent_tversky_alpha=0.3, intent_tversky_beta=0.7,
    )
    decoder = VLMIntentDecoder(config).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    start_epoch, best_val_loss = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        decoder.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
    if world_size > 1:
        decoder = DDP(decoder, device_ids=[local_rank])
    core = decoder.module if world_size > 1 else decoder
    if rank == 0:
        args.logdir.mkdir(parents=True, exist_ok=True)
        print(f"online intent: train={len(train_ds)} val={len(val_ds)} "
              f"world_size={world_size} decoder_batch/gpu={per_gpu_batch} "
              f"qwen_microbatch={args.qwen_batch_size}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        decoder.train()
        train_sum = train_count = 0
        for images, labels in tqdm(train_loader, desc=f"train {epoch}", disable=rank != 0):
            hidden = qwen_image_tokens(
                images, qwen, processor, image_token_id, merge,
                args.qwen_batch_size, device,
            )
            prediction = decoder(hidden)
            losses, logs = {}, {}
            core.compute_loss(prediction, {"visual_intent_label": labels.to(device)}, losses, logs)
            loss = losses["loss_visual_intent"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * len(images)
            train_count += len(images)
        train_loss = reduce_average(train_sum, train_count, device)

        decoder.eval()
        val_sum = val_count = 0
        with torch.inference_mode():
            for images, labels in tqdm(val_loader, desc=f"val {epoch}", disable=rank != 0):
                hidden = qwen_image_tokens(
                    images, qwen, processor, image_token_id, merge,
                    args.qwen_batch_size, device,
                )
                prediction = decoder(hidden)
                losses, logs = {}, {}
                core.compute_loss(prediction, {"visual_intent_label": labels.to(device)}, losses, logs)
                val_sum += losses["loss_visual_intent"].item() * len(images)
                val_count += len(images)
        val_loss = reduce_average(val_sum, val_count, device)
        if rank == 0:
            best_val_loss = min(best_val_loss, val_loss)
            checkpoint = dict(
                epoch=epoch, model=core.state_dict(), optimizer=optimizer.state_dict(),
                loss=train_loss, val_loss=val_loss, best_val_loss=best_val_loss,
            )
            path = args.logdir / f"model_{epoch:04d}.pth"
            torch.save(checkpoint, str(path) + ".tmp")
            os.replace(str(path) + ".tmp", path)
            if val_loss <= best_val_loss:
                best = args.logdir / "model_best.pth"
                torch.save(checkpoint, str(best) + ".tmp")
                os.replace(str(best) + ".tmp", best)
            print(f"epoch={epoch} train={train_loss:.4f} val={val_loss:.4f} "
                  f"best={best_val_loss:.4f}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
