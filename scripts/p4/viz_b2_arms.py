"""P5 B2 go/no-go: visualize the K route arms the multimodal planner emits at junctions.

The whole point of B2 is that the planner emits DISTINCT arms (straight / left / right)
rather than collapsing to one line. This loads the B2 checkpoint, runs the full multimodal
forward on junction frames, and draws the K route arms (coloured, valid arms only) over the
GT drivable blob + the anchor directions, so we can eyeball:
  - do the K arms actually diverge (not overlapping)?
  - does each arm follow its anchor direction / a real blob branch?
  - winner (nearest GT) vs the rest.

Uses the training dataset path (CARLAData with vlm_hidden + anchor) so status/anchor formats
are correct. Runs the canonical TFv6 forward; K arms come back in data["route_multimodal"].

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/p4/viz_b2_arms.py \
        --ckpt-dir outputs/local_training/p5_stepB2 --n 12
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


# BGR-ish distinct colors per mode (drawn on RGB then written; we invert at save)
_MODE_COLORS = [
    (255, 60, 60), (60, 255, 60), (80, 160, 255), (255, 220, 40),
    (255, 100, 255), (60, 255, 255),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="outputs/local_training/p5_stepB2")
    ap.add_argument("--ckpt-name", default="model_0019.pth")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="outputs/viz_b2_arms")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import cv2
    import timm
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.tfv6.tfv6 import TFv6
    from lead.training.config_training import TrainingConfig
    from lead.training.mixed_training_utils import mixed_data_collate_fn

    device = torch.device("cuda:0")
    with open(os.path.join(args.ckpt_dir, "config.json")) as f:
        config = TrainingConfig(json.load(f), raise_error_on_missing_key=False)
    _c = timm.create_model
    timm.create_model = lambda *a, **k: _c(*a, **{**k, "pretrained": False})

    model = TFv6(device, config).to(device)
    model.load_state_dict(
        torch.load(os.path.join(args.ckpt_dir, args.ckpt_name), map_location=device, weights_only=True),
        strict=True,
    )
    model.eval().requires_grad_(False)

    ppm = config.pixels_per_meter
    row_ego = int((0 - config.min_y_meter) * ppm)  # 160
    col_ego = int((0 - config.min_x_meter) * ppm)  # 128

    # junction frames
    cmds = {(e["scenario"], e["route"], e["frame"]): e.get("command")
            for e in (json.loads(l) for l in open(config.vlm_manifest))}

    carla_ds = CARLAData(root=config.carla_data, config=config)
    vlm_ds = VLMIntentDataset(carla_ds, vlm_cache_dir=config.vlm_cache_dir,
                              manifest_path=config.vlm_manifest)

    def m2px(xy):
        # xy: (...,2) ego metres [x=long(forward), y=lat(right)] -> (row,col)
        x, y = xy[..., 0], xy[..., 1]
        col = col_ego + x * ppm
        row = row_ego + y * ppm
        return row, col

    saved = 0
    for idx in range(len(vlm_ds)):
        if saved >= args.n:
            break
        ci = vlm_ds.valid_indices[idx]
        p = str(carla_ds.images[ci], encoding="utf-8").split("/")
        srf = (p[-4], p[-3], p[-1].split(".")[0])
        if cmds.get(srf) in (None, "LANEFOLLOW"):
            continue
        data = vlm_ds[idx]
        batch = mixed_data_collate_fn([data])
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=config.torch_float_type):
            model(batch)  # fills batch["route_multimodal"], batch["route_conf"]
        if "route_multimodal" not in batch:
            print("no route_multimodal -- is multimodal_planner set in config?")
            return
        arms = batch["route_multimodal"][0].float().cpu().numpy()  # (K,n,2)
        conf = torch.sigmoid(batch["route_conf"][0].float()).cpu().numpy()  # (K,)
        anchor = np.asarray(data["anchor"])  # (K,4) [sin,cos,reach/30,valid]
        valid = anchor[:, 3] > 0.5

        # canvas: GT blob (grey) as background
        gt = data.get("visual_intent_label")
        gt = np.asarray(gt)[0] if gt is not None else np.zeros((config.lidar_height_pixel, config.lidar_width_pixel))
        canvas = (np.clip(gt, 0, 1)[..., None] * np.array([90, 90, 90])).astype(np.uint8)
        cv2.circle(canvas, (col_ego, row_ego), 3, (255, 255, 255), -1)

        nvalid = int(valid.sum())
        for k in range(arms.shape[0]):
            if not valid[k]:
                continue
            r, c = m2px(arms[k])
            pts = np.stack([c, r], axis=1).astype(np.int32)
            col_k = _MODE_COLORS[k % len(_MODE_COLORS)]
            cv2.polylines(canvas, [pts], False, col_k, 2)
            cv2.putText(canvas, f"{conf[k]:.2f}", (int(c[-1]), int(r[-1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col_k, 1)

        # forward = +col = image right; rotate 90 CCW so forward is UP
        canvas = cv2.rotate(canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.putText(canvas, f"K={nvalid} {srf[0][:18]} cmd={cmds.get(srf)}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        outp = os.path.join(args.out, f"{saved:02d}_K{nvalid}_idx{idx}.png")
        cv2.imwrite(outp, canvas[..., ::-1])
        saved += 1
        print(f"saved {outp}  {srf}  K={nvalid}  conf={conf[valid].round(2)}")

    print(f"\n✓ {saved} panels in {args.out}")


if __name__ == "__main__":
    main()
