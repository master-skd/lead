"""Compare a live Qwen service response with a saved three-camera training feature."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lead.inference.vlm_client import VLMServiceClient


def feature_metrics(reference: np.ndarray, live: np.ndarray) -> dict:
    if reference.shape != live.shape:
        return {"shape_match": False, "reference_shape": reference.shape, "live_shape": live.shape}
    a = np.asarray(reference, dtype=np.float32).reshape(-1)
    b = np.asarray(live, dtype=np.float32).reshape(-1)
    delta = np.abs(a - b)
    return {
        "shape_match": True,
        "mean_abs_error": float(delta.mean()),
        "p99_abs_error": float(np.quantile(delta, 0.99)),
        "max_abs_error": float(delta.max()),
        "cosine_similarity": float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)),
        "fraction_exact": float((a == b).mean()),
        "reference_mean_std": (float(a.mean()), float(a.std())),
        "live_mean_std": (float(b.mean()), float(b.std())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Original full three-camera training JPEG")
    parser.add_argument("--cache", type=Path, required=True, help="Matching cached Qwen hidden .npy")
    parser.add_argument("--socket", default="/tmp/vlm_service.sock")
    parser.add_argument("--check-bgr", action="store_true", help="Also send channel-reversed image")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    cache = np.load(args.cache, allow_pickle=False)
    print(f"image RGB shape={image.shape}; cached feature shape={cache.shape}", flush=True)
    client = VLMServiceClient(args.socket)
    try:
        live = client.extract(image)
        print(f"RGB request: {feature_metrics(cache, live)}", flush=True)
        if args.check_bgr:
            live_bgr = client.extract(image[..., ::-1])
            print(f"channel-reversed request: {feature_metrics(cache, live_bgr)}", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
