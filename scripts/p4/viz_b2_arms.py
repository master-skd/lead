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

# ChaffeurNet BEV semantic class -> RGB (road grey, sidewalk brown, lane markers yellow)
_SEM_RGB = {
    0: (0, 0, 0),        # unlabeled
    1: (70, 70, 70),     # road
    2: (110, 80, 50),    # sidewalk
    3: (200, 200, 0),    # lane markers
    4: (140, 140, 0),    # lane markers broken
    5: (200, 0, 0),      # stop signs
    6: (0, 150, 0),      # traffic green
    7: (180, 180, 0),    # traffic yellow
    8: (150, 0, 0),      # traffic red
}


def _hdmap_bev(hdmap_path, config):
    """Colorized raw hdmap semantics resampled to the ego BEV grid (same transform as
    rasterize_reachable_support), so trajectories can be judged against the actual road."""
    import cv2
    import numpy as np
    hdmap = cv2.imread(hdmap_path, cv2.IMREAD_UNCHANGED)
    if hdmap is not None and hdmap.ndim == 3:
        hdmap = hdmap[..., 0]
    h_px, w_px = config.lidar_height_pixel, config.lidar_width_pixel
    ppm, ppm_c = config.pixels_per_meter, config.pixels_per_meter_collection
    canvas = np.zeros((h_px, w_px, 3), dtype=np.uint8)
    if hdmap is None:
        return canvas
    rows, cols = np.meshgrid(np.arange(h_px), np.arange(w_px), indexing="ij")
    x_m = cols / ppm + config.min_x_meter
    y_m = rows / ppm + config.min_y_meter
    hx = np.round(hdmap.shape[1] / 2 + x_m * ppm_c).astype(np.int64)
    hy = np.round(hdmap.shape[0] / 2 + y_m * ppm_c).astype(np.int64)
    inside = (hx >= 0) & (hx < hdmap.shape[1]) & (hy >= 0) & (hy < hdmap.shape[0])
    sem = np.zeros((h_px, w_px), dtype=np.uint8)
    sem[inside] = hdmap[hy[inside], hx[inside]]
    for cls, rgb in _SEM_RGB.items():
        canvas[sem == cls] = rgb
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="outputs/local_training/p5_stepB2")
    ap.add_argument("--ckpt-name", default="model_0019.pth")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="outputs/viz_b2_arms")
    ap.add_argument("--show-padding", action="store_true",
                    help="also draw the INVALID (padding) arms in dim grey. They get no "
                         "gradient from any loss, so they drift freely -- worth seeing "
                         "because closed-loop arm selection must mask them out.")
    ap.add_argument("--min-arms", type=int, default=1,
                    help="only save frames with at least this many valid arms (use 2 to "
                         "see actual forks rather than the 80%% single-arm majority)")
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
        if int(valid.sum()) < args.min_arms:
            continue
        gt = np.asarray(data["route"], dtype=np.float32)[:, :2]  # (n,2) expert route

        # canvas: colorized hdmap semantics (road / sidewalk / lane markers) resampled to
        # the ego BEV grid, so we can see if each arm stays ON THE ROAD (a divergent arm is
        # only useful if it follows a real drivable branch, not grass/oncoming/curb).
        canvas = _hdmap_bev(str(carla_ds.bev_semantics[ci], encoding="utf-8"), config)
        cv2.circle(canvas, (col_ego, row_ego), 3, (255, 255, 255), -1)

        # GT expert route first (thick white), so arms are judged against it
        r, c = m2px(gt)
        cv2.polylines(canvas, [np.stack([c, r], 1).astype(np.int32)], False, (255, 255, 255), 3)

        nvalid = int(valid.sum())
        # padding arms underneath (dim grey, thin) -- they are unconstrained by every loss
        if args.show_padding:
            for k in range(arms.shape[0]):
                if valid[k]:
                    continue
                r, c = m2px(arms[k])
                cv2.polylines(canvas, [np.stack([c, r], 1).astype(np.int32)], False,
                              (90, 90, 90), 1)

        # winner = valid arm closest to GT, marked so WTA behaviour is visible
        d_gt = np.linalg.norm(arms - gt[None], axis=-1).mean(axis=1)
        d_gt[~valid] = np.inf
        winner = int(np.argmin(d_gt))

        for k in range(arms.shape[0]):
            if not valid[k]:
                continue
            r, c = m2px(arms[k])
            pts = np.stack([c, r], axis=1).astype(np.int32)
            col_k = _MODE_COLORS[k % len(_MODE_COLORS)]
            cv2.polylines(canvas, [pts], False, col_k, 3 if k == winner else 2)
            tag = f"{conf[k]:.2f}" + ("*" if k == winner else "")
            cv2.putText(canvas, tag, (int(c[-1]), int(r[-1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col_k, 1)

        # forward = +col = image right; rotate 90 CCW so forward is UP
        canvas = cv2.rotate(canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.putText(canvas, f"K={nvalid} {srf[0][:18]} cmd={cmds.get(srf)}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(canvas, "white=GT  thick=winner  grey=padding(unconstrained)",
                    (5, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        outp = os.path.join(args.out, f"{saved:02d}_K{nvalid}_idx{idx}.png")
        cv2.imwrite(outp, canvas[..., ::-1])
        saved += 1
        spread = float(np.linalg.norm(
            arms[valid][:, -1][:, None] - arms[valid][:, -1][None], axis=-1,
        ).max()) if nvalid > 1 else 0.0
        print(f"saved {outp}  {srf[0][:22]}  K={nvalid}  conf={conf[valid].round(2)}  "
              f"max_endpoint_spread={spread:.1f}m")

    print(f"\n✓ {saved} panels in {args.out}")


if __name__ == "__main__":
    main()
