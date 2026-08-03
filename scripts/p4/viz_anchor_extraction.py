"""P5 B2a: visualize anchor extraction on the predicted drivable blob.

Runs the frozen P5a intent head on junction-biased frames, extracts K directional arms
via anchor_extraction.extract_anchors_from_blob, and saves 3-panel PNGs:
    [GT blob] | [pred blob] | [pred blob + sector rays + extracted anchor tips]

Verify: junctions -> 3-4 arms in the right directions; straight road -> 1 forward arm;
anchors land on sensible blob endpoints; robust to the mushy predicted blob.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/p4/viz_anchor_extraction.py \
        --ckpt outputs/local_training/vlm_intent_p5a_tversky/model_0014.pth --n 10
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


def _heat(h):
    h = np.clip(h, 0, 1)
    g = np.where(h < 0.5, h * 2, 1.0 - (h - 0.5) * 2)
    return (np.stack([np.clip(h * 2, 0, 1), g, np.zeros_like(h)], -1) * 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/local_training/vlm_intent_p5a_tversky/model_0014.pth")
    ap.add_argument("--cache", default="data/p5/vlm_cache")
    ap.add_argument("--manifest", default="data/p4/manifest.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="outputs/viz_anchor")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import cv2
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.tfv6.vlm_intent_decoder import VLMIntentDecoder
    from lead.tfv6.anchor_extraction import extract_anchors_from_blob, K_MAX
    from lead.training.config_training import TrainingConfig

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = TrainingConfig(
        {
            "use_intent_decoder": True, "use_multimodal_intent": True,
            "multimodal_intent_horizon_m": 30.0, "vlm_cache_dir": args.cache,
            "vlm_manifest": args.manifest, "carla_data": "data/carla_leaderboard2/data",
        },
        raise_error_on_missing_key=False,
    )
    ppm = cfg.pixels_per_meter
    row_ego = int((0 - cfg.min_y_meter) * ppm)   # 160
    col_ego = int((0 - cfg.min_x_meter) * ppm)   # 128

    entries = [json.loads(l) for l in open(args.manifest)]
    junction_srf = {
        (e["scenario"], e["route"], e["frame"])
        for e in entries if e.get("command") not in (None, "LANEFOLLOW")
    }

    decoder = VLMIntentDecoder(cfg).to(device)
    decoder.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True)["model"])
    decoder.eval().requires_grad_(False)

    carla_ds = CARLAData(root=cfg.carla_data, config=cfg)
    vlm_ds = VLMIntentDataset(carla_ds, vlm_cache_dir=args.cache, manifest_path=args.manifest)

    def _srf(ci):
        p = str(carla_ds.images[ci], encoding="utf-8").split("/")
        return (p[-4], p[-3], p[-1].split(".")[0])

    saved = 0
    for idx in range(len(vlm_ds)):
        if saved >= args.n:
            break
        ci = vlm_ds.valid_indices[idx]
        srf = _srf(ci)
        if srf not in junction_srf:
            continue
        data = vlm_ds[idx]
        vh = data["vlm_hidden"].to(device).float().unsqueeze(0)
        with torch.no_grad():
            pred = torch.sigmoid(decoder(vh)).cpu().numpy()[0, 0]
        gt = data.get("visual_intent_label")
        gt = np.asarray(gt)[0] if gt is not None else np.zeros_like(pred)

        anchors, endpoints = extract_anchors_from_blob(pred, ppm, row_ego, col_ego)
        n_arms = int(anchors[:, 0].sum())

        p_gt = _heat(gt)
        p_pred = _heat(pred)
        p_anno = _heat(pred).copy()
        # draw ego
        cv2.circle(p_anno, (col_ego, row_ego), 3, (0, 255, 255), -1)
        # draw each valid arm: ray + tip
        for k in range(K_MAX):
            if anchors[k, 0] < 0.5:
                continue
            r, c = int(endpoints[k, 0]), int(endpoints[k, 1])
            cv2.line(p_anno, (col_ego, row_ego), (c, r), (0, 255, 255), 1)
            cv2.circle(p_anno, (c, r), 4, (255, 255, 255), -1)

        # Rotate all panels so forward (+col) points UP (intuitive "car drives up").
        # +col is image-right; rotate 90 deg CCW -> +col becomes up.
        rot = lambda im: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        p_gt, p_pred, p_anno = rot(p_gt), rot(p_pred), rot(p_anno)
        # thin white separators between panels
        sep = np.full((p_gt.shape[0], 2, 3), 255, np.uint8)
        panel = np.concatenate([p_gt, sep, p_pred, sep, p_anno], axis=1)
        cv2.putText(panel, f"GT | pred | anchors K={n_arms} (fwd=UP) {srf[0][:16]}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        out = os.path.join(args.out, f"{saved:02d}_K{n_arms}_idx{idx}.png")
        cv2.imwrite(out, panel[..., ::-1])
        saved += 1
        print(f"saved {out}  {srf}  K_scene={n_arms}  angles={np.degrees(anchors[anchors[:,0]>0.5,1]).round(0)}")

    print(f"\n✓ {saved} panels in {args.out}")


if __name__ == "__main__":
    main()
