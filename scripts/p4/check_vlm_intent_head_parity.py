"""Compare cached and live Qwen features after the frozen VLM intent head.

Run from the lead environment while the Qwen services run separately in their
respective qwenvl environments. No CARLA process is needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lead.inference.vlm_client import VLMServiceClient
from lead.tfv6.vlm_intent_decoder import VLMIntentDecoder
from lead.training.config_training import TrainingConfig


SAMPLE = Path("OppositeVehicleTakingPriority/Town12_Rep0_1900_0_route0_01_08_20_31_40/0008")


def predict_intent(model: VLMIntentDecoder, hidden: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        tensor = torch.from_numpy(np.asarray(hidden, dtype=np.float32))[None]
        return torch.sigmoid(model(tensor))[0, 0].numpy()


def summarize(name: str, probability: np.ndarray, reference: np.ndarray | None) -> None:
    positive = probability > 0.5
    fields = [
        f"{name}: ego_prob={probability[160, 128]:.4f}",
        f"near_positive={int(positive[152:169, 120:137].sum())}",
        f"total_positive={int(positive.sum())}",
    ]
    if reference is not None:
        reference_positive = reference > 0.5
        union = (positive | reference_positive).sum()
        overlap = (positive & reference_positive).sum()
        fields.extend([
            f"mask_IoU_vs_cache={overlap / max(int(union), 1):.4f}",
            f"prob_MAE_vs_cache={np.abs(probability - reference).mean():.4f}",
        ])
    print(" ".join(fields), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=Path("/tmp/vlm_parity_sample"))
    parser.add_argument(
        "--head", type=Path,
        default=Path("outputs/local_training/vlm_intent_p6_lanegraph/model_0014.pth"),
    )
    parser.add_argument("--old-socket", default="/tmp/vlm_service.sock")
    parser.add_argument("--new-socket", default="/tmp/vlm_service_v513.sock")
    parser.add_argument("--skip-old", action="store_true")
    args = parser.parse_args()

    image_path = (
        args.sample_dir / "data/carla_leaderboard2/data" / SAMPLE.parent / "rgb" /
        f"{SAMPLE.name}.jpg"
    )
    cache_path = args.sample_dir / "data/p6/vlm_cache_3cam" / SAMPLE.with_suffix(".npy")
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    cache = np.load(cache_path, allow_pickle=False)
    print(f"image={image.shape} cached_hidden={cache.shape}", flush=True)

    torch.set_num_threads(4)
    model = VLMIntentDecoder(TrainingConfig()).eval()
    checkpoint = torch.load(args.head, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    reference = predict_intent(model, cache)
    summarize("cache", reference, None)

    sockets = [("v513", args.new_socket)]
    if not args.skip_old:
        sockets.insert(0, ("old", args.old_socket))
    for name, socket_path in sockets:
        client = VLMServiceClient(socket_path)
        try:
            hidden = client.extract(image)
        finally:
            client.close()
        if hidden.shape != cache.shape:
            print(f"{name}: hidden shape {hidden.shape} != cached {cache.shape}", flush=True)
            continue
        probability = predict_intent(model, hidden)
        summarize(name, probability, reference)


if __name__ == "__main__":
    main()
