"""Visualize the P5a multimodal (drivable-blob) VLM intent head.

Loads the frozen P5a VLMIntentDecoder, runs it on cached vlm_hidden for a handful of
frames (biased toward junctions), and saves 2-panel PNGs:
    [GT drivable-support blob] | [P5a predicted intent]
both as heatmaps. Point: does the prediction light up ALL feasible arms at a junction
(the multimodal claim), or does it collapse to a single blob / go mushy?

Junction bias: picks frames whose manifest command != LANEFOLLOW (turns / lane changes),
where multi-arm structure should be visible.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/p4/viz_p5a_intent.py \
        --ckpt outputs/local_training/vlm_intent_p5a_tversky/model_0014.pth --n 8
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


def _heat_to_rgb(heat: np.ndarray) -> np.ndarray:
    """Map a [0,1] field to a simple black->yellow->red heatmap (no matplotlib dep)."""
    h = np.clip(heat, 0, 1)
    r = np.clip(h * 2, 0, 1)
    g = np.clip(h * 2 - 0.0, 0, 1) * (h < 0.5) + (1.0 - (h - 0.5) * 2) * (h >= 0.5)
    b = np.zeros_like(h)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/local_training/vlm_intent_p5a_tversky/model_0014.pth")
    ap.add_argument("--cache", default="data/p5/vlm_cache")
    ap.add_argument("--manifest", default="data/p4/manifest.jsonl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="outputs/viz_p5a_intent")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import cv2
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.tfv6.vlm_intent_decoder import VLMIntentDecoder
    from lead.training.config_training import TrainingConfig

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # multimodal intent label = drivable blob (needs the flag so CARLAData rasterizes it)
    cfg = TrainingConfig(
        {
            "use_intent_decoder": True,  # gate that triggers label rasterization in CARLAData
            "use_multimodal_intent": True,
            "multimodal_intent_horizon_m": 30.0,
            "vlm_cache_dir": args.cache,
            "vlm_manifest": args.manifest,
            "carla_data": "data/carla_leaderboard2/data",
        },
        raise_error_on_missing_key=False,
    )

    # pick junction-ish frames (command != LANEFOLLOW). Command lookup always uses the
    # fixed p4 manifest (independent of which cache we filter frames by).
    cmd_manifest = "data/p4/manifest.jsonl"
    entries = [json.loads(l) for l in open(cmd_manifest)]
    junction_srf = {
        (e["scenario"], e["route"], e["frame"])
        for e in entries if e.get("command") not in (None, "LANEFOLLOW")
    }
    print(f"{len(junction_srf)} junction frames in manifest")

    decoder = VLMIntentDecoder(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    decoder.load_state_dict(ckpt["model"])
    decoder.eval().requires_grad_(False)

    carla_ds = CARLAData(root=cfg.carla_data, config=cfg)
    # manifest="none" -> filter frames by actual .npy existence (P6 4000-frame subset).
    mpath = None if args.manifest.lower() == "none" else args.manifest
    vlm_ds = VLMIntentDataset(carla_ds, vlm_cache_dir=args.cache, manifest_path=mpath)

    def _srf(carla_idx):
        p = str(carla_ds.images[carla_idx], encoding="utf-8").split("/")
        return (p[-4], p[-3], p[-1].split(".")[0])

    saved = 0
    for idx in range(len(vlm_ds)):
        if saved >= args.n:
            break
        carla_idx = vlm_ds.valid_indices[idx]
        srf = _srf(carla_idx)
        if srf not in junction_srf:  # junction frames only
            continue
        data = vlm_ds[idx]
        vlm_hidden = data["vlm_hidden"].to(device).float().unsqueeze(0)
        with torch.no_grad():
            pred = torch.sigmoid(decoder(vlm_hidden)).cpu().numpy()[0, 0]  # (H,W)
        gt = data.get("visual_intent_label")
        gt = np.asarray(gt)[0] if gt is not None else np.zeros_like(pred)

        panel = np.concatenate([_heat_to_rgb(gt), _heat_to_rgb(pred)], axis=1)
        cv2.putText(panel, f"GT blob {srf[0][:14]}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(panel, "P5a pred", (pred.shape[1] + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        out_path = os.path.join(args.out, f"{saved:02d}_idx{idx}.png")
        cv2.imwrite(out_path, panel[..., ::-1])
        saved += 1
        print(f"saved {out_path}  {srf}  pred[max={pred.max():.3f} mean={pred.mean():.3f}] gt[max={gt.max():.3f} mean={gt.mean():.3f}]")

    print(f"\n✓ {saved} panels in {args.out}")


if __name__ == "__main__":
    main()
