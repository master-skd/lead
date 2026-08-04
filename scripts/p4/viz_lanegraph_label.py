"""P5 B2: visualize lane-graph arms AS AN INTENT SUPERVISION LABEL, vs the current mushy
blob label. 3-panel per frame: [hdmap] | [current blob label] | [lane-graph arm label].

The lane-graph label is built by expanding the lane graph forward (all branches) from ego,
rasterizing each arm centerline as a soft Gaussian corridor into the ego BEV grid -- a
multi-arm, on-lane, direction-aware target, unlike the flood-fill blob that mushes all
reachable road into one region.
"""
from __future__ import annotations
import json, sys
import numpy as np

HORIZON, STEP, SIGMA_PX = 40.0, 1.0, 3.0


def main():
    import carla, cv2, os
    from scipy import ndimage
    from lead.common import common_utils
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.tfv6.intent_decoder import rasterize_reachable_support
    from lead.training.config_training import TrainingConfig
    sys.path.insert(0, "scripts/p4")
    from viz_b2_arms import _hdmap_bev

    cfg = TrainingConfig(json.load(open("outputs/local_training/p5_stepB2_v2/config.json")),
                         raise_error_on_missing_key=False)
    cfg_mm = TrainingConfig({"use_multimodal_intent": True, "multimodal_intent_horizon_m": 30.0},
                            raise_error_on_missing_key=False)
    cds = CARLAData(root=cfg.carla_data, config=cfg)
    vds = VLMIntentDataset(cds, vlm_cache_dir=cfg.vlm_cache_dir, manifest_path=cfg.vlm_manifest)
    ppm = cfg.pixels_per_meter
    H, W = cfg.lidar_height_pixel, cfg.lidar_width_pixel
    row_ego = int((0 - cfg.min_y_meter) * ppm); col_ego = int((0 - cfg.min_x_meter) * ppm)

    cmap_cache = {}
    def get_map(town):
        if town not in cmap_cache:
            import glob
            cands = [
                f"3rd_party/CARLA_0915/CarlaUE4/Content/Carla/Maps/{town}/OpenDrive/{town}.xodr",
                f"3rd_party/CARLA_0915/CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr",
            ]
            xodr = next((c for c in cands if os.path.exists(c)), None)
            if xodr is None:
                hits = glob.glob(f"3rd_party/CARLA_0915/**/{town}.xodr", recursive=True)
                xodr = hits[0] if hits else None
            cmap_cache[town] = carla.Map(town, open(xodr).read())
        return cmap_cache[town]

    def heat(x):
        h = np.clip(x, 0, 1)
        return (np.stack([np.clip(h*2,0,1), np.where(h<.5,h*2,1-(h-.5)*2), np.zeros_like(h)],-1)*255).astype(np.uint8)

    IDXS = [9, 26, 40, 68, 70]  # junction frames incl. the bad idx68
    import os; os.makedirs("outputs/viz_lanegraph_label", exist_ok=True)
    for saved, IDX in enumerate(IDXS):
        ci = vds.valid_indices[IDX]
        base = "/".join(str(cds.images[ci], encoding="utf-8").split("/")[:-2])
        frame = str(cds.images[ci], encoding="utf-8").split("/")[-1].split(".")[0]
        m = common_utils.read_pickle(f"{base}/metas/{frame}.pkl")
        town, pos, ego_mat = m["town"], m["pos_global"], np.array(m["ego_matrix"])
        cmap = get_map(town)
        wp0 = cmap.get_waypoint(carla.Location(x=pos[0], y=pos[1], z=pos[2]))

        arms = []
        def walk(wp, acc, dist):
            acc = acc + [(wp.transform.location.x, wp.transform.location.y, wp.transform.location.z)]
            if dist >= HORIZON:
                arms.append(acc); return
            nxts = wp.next(STEP)
            if not nxts:
                arms.append(acc); return
            if len(nxts) == 1:
                walk(nxts[0], acc, dist + STEP)
            else:
                for nx in nxts:
                    walk(nx, acc, dist + STEP)
        walk(wp0, [], 0.0)

        # rasterize arms -> soft label
        inv = np.linalg.inv(ego_mat)
        label = np.zeros((H, W), np.float32)
        for arm in arms:
            for (wx, wy, wz) in arm:
                v = inv @ np.array([wx, wy, wz, 1.0])
                col = int(round(col_ego + v[0] * ppm)); row = int(round(row_ego + v[1] * ppm))
                if 0 <= row < H and 0 <= col < W:
                    label[row, col] = 1.0
        label = ndimage.gaussian_filter(label, SIGMA_PX)
        if label.max() > 0: label /= label.max()

        # current blob label
        raw_hdmap = cv2.imread(str(cds.bev_semantics[ci], encoding="utf-8"), cv2.IMREAD_UNCHANGED)
        blob = rasterize_reachable_support(raw_hdmap, cfg_mm, horizon_m=30.0)[0]

        hd = _hdmap_bev(str(cds.bev_semantics[ci], encoding="utf-8"), cfg)
        sep = np.full((H, 2, 3), 255, np.uint8)
        combined = np.concatenate([hd, sep, heat(blob), sep, heat(label)], axis=1)
        panel = cv2.rotate(combined, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.putText(panel, f"idx{IDX} {town} | hdmap | blob-label | lanegraph-label({len(arms)}arms) fwdUP",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        out = f"outputs/viz_lanegraph_label/{saved:02d}_idx{IDX}.png"
        cv2.imwrite(out, panel[..., ::-1])
        print(f"saved {out}  {town}  arms={len(arms)}")


if __name__ == "__main__":
    main()
