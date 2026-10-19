#!/usr/bin/env python3
"""Inspect one closed-loop P6 intent capture and optionally replay each stage."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np


def compare_arrays(reference: np.ndarray, replayed: np.ndarray) -> tuple[float, float, float]:
    """Return MAE, maximum absolute error, and cosine similarity."""
    if reference.shape != replayed.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {replayed.shape}")
    left = reference.astype(np.float32).reshape(-1)
    right = replayed.astype(np.float32).reshape(-1)
    difference = np.abs(left - right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    cosine = float(np.dot(left, right) / denominator) if denominator else float("nan")
    return float(difference.mean()), float(difference.max()), cosine


def describe_intent(logits: np.ndarray) -> dict[str, float | int | None]:
    """Report calibration and anchor extraction, not just an image-wide mean."""
    from lead.tfv6.anchor_extraction import extract_anchors_from_blob

    probability = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float32), -80, 80)))
    try:
        anchors, _ = extract_anchors_from_blob(probability, 4.0, 160, 128)
        valid_anchors = int(anchors[:, 0].sum())
    except ModuleNotFoundError as exc:
        if exc.name != "scipy":
            raise
        valid_anchors = None
    return {
        "prob_max": float(probability.max()),
        "prob_p99": float(np.quantile(probability, 0.99)),
        "prob_p999": float(np.quantile(probability, 0.999)),
        "coverage_0.25": float((probability > 0.25).mean()),
        "coverage_0.35": float((probability > 0.35).mean()),
        "coverage_0.50": float((probability > 0.50).mean()),
        "coverage_0.65": float((probability > 0.65).mean()),
        "valid_anchors_0.50": valid_anchors,
    }


def jpeg_roundtrip(rgb: np.ndarray, quality: int):
    """Approximate the training data's stitched-strip JPEG save/read boundary."""
    from PIL import Image

    image = Image.fromarray(rgb, "RGB")
    encoded = BytesIO()
    image.save(encoded, format="JPEG", quality=quality)
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        return decoded.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--intent-checkpoint", type=Path,
                        help="replay frozen intent decoder on the captured Qwen hidden")
    parser.add_argument("--qwen-model", type=Path,
                        help="re-extract Qwen hidden from captured RGB using this Python environment")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--jpeg-quality", type=int, action="append", default=[],
                        help="also replay a stitched-strip JPEG round trip at this quality; repeatable")
    args = parser.parse_args()
    if any(not 1 <= quality <= 100 for quality in args.jpeg_quality):
        parser.error("--jpeg-quality must be in [1, 100]")
    if args.jpeg_quality and args.qwen_model is None:
        parser.error("--jpeg-quality requires --qwen-model")

    with np.load(args.capture, allow_pickle=False) as archive:
        rgb = archive["rgb"]
        hidden = archive["vlm_hidden"]
        logits = archive["intent_logits"]
        step = int(archive["step"])
    print(f"step={step} rgb={rgb.shape}/{rgb.dtype} "
          f"hidden={hidden.shape}/{hidden.dtype} logits={logits.shape}/{logits.dtype}")
    print("captured intent:", describe_intent(logits))

    intent_model = None
    if args.intent_checkpoint is not None:
        import torch
        from types import SimpleNamespace

        from lead.tfv6.vlm_intent_decoder import VLMIntentDecoder

        config = SimpleNamespace(lidar_height_pixel=320, lidar_width_pixel=384)
        intent_model = VLMIntentDecoder(config).eval()
        checkpoint = torch.load(args.intent_checkpoint, map_location="cpu", weights_only=True)
        intent_model.load_state_dict(checkpoint["model"])
        with torch.inference_mode():
            replayed = intent_model(torch.from_numpy(hidden.astype(np.float32))[None])[0, 0].numpy()
        print("decoder replay MAE/max/cosine:", compare_arrays(logits, replayed))
        print("replayed intent:", describe_intent(replayed))

    if args.qwen_model is not None:
        from PIL import Image
        import transformers

        from lead.inference.vlm_feature_extractor import (
            extract_vlm_hidden,
            load_qwen_vl,
        )

        model, processor, image_token_id, merge = load_qwen_vl(
            str(args.qwen_model), device=args.device,
        )
        print(f"Qwen replay environment: transformers={transformers.__version__} "
              f"device={args.device}")
        variants = [("raw", Image.fromarray(rgb, "RGB"))]
        variants.extend(
            (f"jpeg_q{quality}", jpeg_roundtrip(rgb, quality))
            for quality in args.jpeg_quality
        )
        raw_hidden = None
        for name, image in variants:
            if name != "raw":
                pixels = np.asarray(image, dtype=np.uint8)
                pixel_mae = np.abs(rgb.astype(np.int16) - pixels.astype(np.int16)).mean()
                print(f"{name} pixel MAE vs captured raw RGB: {pixel_mae:.6f}")
            replayed = extract_vlm_hidden(
                image, model, processor, image_token_id, merge,
                prompt_mode="3cam_drivable", device=args.device,
            )
            print(f"{name} Qwen vs capture MAE/max/cosine:",
                  compare_arrays(hidden, replayed))
            if raw_hidden is None:
                raw_hidden = replayed
            else:
                print(f"{name} Qwen vs raw MAE/max/cosine:",
                      compare_arrays(raw_hidden, replayed))
            if intent_model is not None:
                import torch

                with torch.inference_mode():
                    replayed_logits = intent_model(
                        torch.from_numpy(replayed.astype(np.float32))[None]
                    )[0, 0].numpy()
                print(f"{name} intent vs capture MAE/max/cosine:",
                      compare_arrays(logits, replayed_logits))
                print(f"{name} intent:", describe_intent(replayed_logits))


if __name__ == "__main__":
    main()
