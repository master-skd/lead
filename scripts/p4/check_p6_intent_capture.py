#!/usr/bin/env python3
"""Inspect one closed-loop P6 intent capture and optionally replay each stage."""

from __future__ import annotations

import argparse
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


def describe_intent(logits: np.ndarray) -> dict[str, float | int]:
    """Report calibration and anchor extraction, not just an image-wide mean."""
    from lead.tfv6.anchor_extraction import extract_anchors_from_blob

    probability = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float32), -80, 80)))
    anchors, _ = extract_anchors_from_blob(probability, 4.0, 160, 128)
    return {
        "prob_max": float(probability.max()),
        "prob_p99": float(np.quantile(probability, 0.99)),
        "prob_p999": float(np.quantile(probability, 0.999)),
        "coverage_0.25": float((probability > 0.25).mean()),
        "coverage_0.35": float((probability > 0.35).mean()),
        "coverage_0.50": float((probability > 0.50).mean()),
        "coverage_0.65": float((probability > 0.65).mean()),
        "valid_anchors_0.50": int(anchors[:, 0].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--intent-checkpoint", type=Path,
                        help="replay frozen intent decoder on the captured Qwen hidden")
    parser.add_argument("--qwen-model", type=Path,
                        help="re-extract Qwen hidden from captured RGB using this Python environment")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with np.load(args.capture, allow_pickle=False) as archive:
        rgb = archive["rgb"]
        hidden = archive["vlm_hidden"]
        logits = archive["intent_logits"]
        step = int(archive["step"])
    print(f"step={step} rgb={rgb.shape}/{rgb.dtype} "
          f"hidden={hidden.shape}/{hidden.dtype} logits={logits.shape}/{logits.dtype}")
    print("captured intent:", describe_intent(logits))

    if args.intent_checkpoint is not None:
        import torch
        from types import SimpleNamespace

        from lead.tfv6.vlm_intent_decoder import VLMIntentDecoder

        config = SimpleNamespace(lidar_height_pixel=320, lidar_width_pixel=384)
        model = VLMIntentDecoder(config).eval()
        checkpoint = torch.load(args.intent_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        with torch.inference_mode():
            replayed = model(torch.from_numpy(hidden.astype(np.float32))[None])[0, 0].numpy()
        print("decoder replay MAE/max/cosine:", compare_arrays(logits, replayed))
        print("replayed intent:", describe_intent(replayed))

    if args.qwen_model is not None:
        from PIL import Image

        from lead.inference.vlm_feature_extractor import (
            extract_vlm_hidden,
            load_qwen_vl,
        )

        model, processor, image_token_id, merge = load_qwen_vl(
            str(args.qwen_model), device=args.device,
        )
        replayed = extract_vlm_hidden(
            Image.fromarray(rgb, "RGB"), model, processor, image_token_id, merge,
            prompt_mode="3cam_drivable", device=args.device,
        )
        print("Qwen replay MAE/max/cosine:", compare_arrays(hidden, replayed))


if __name__ == "__main__":
    main()
