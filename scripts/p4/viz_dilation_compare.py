"""P5 B2: compare 3 dilation styles for turning lane-graph arms into an intent corridor.
Per frame, 5-panel: [hdmap] | [A gaussian] | [B lane-width] | [C lane-width+gaussian] | [centerlines].
A: thin centerline + gaussian_filter (like the old blob).
B: solid band of lane_width around each centerline.
C: B then gaussian soft edge.
Reuses the L/S/R arm extraction (choose_at_junction) from check_lanegraph_arms.
"""
from __future__ import annotations
import json, os, glob
import numpy as np


def main():
    import carla, cv2
    from scipy import ndimage
    import sys; sys.path.insert(0, "scripts/p4")
    from check_lanegraph_arms import _arm_for_turn
    from viz_b2_arms import _hdmap_bev
    from lead.common import common_utils
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.training.config_training import TrainingConfig

    cfg = TrainingConfig(json.load(open("outputs/local_training/p5_stepB2_v2/config.json")),
                         raise_error_on_missing_key=False)
    cds = CARLAData(root=cfg.carla_data, config=cfg)
    vds = VLMIntentDataset(cds, vlm_cache_dir=cfg.vlm_cache_dir, manifest_path=cfg.vlm_manifest)
    ppm = cfg.pixels_per_meter
    H, W = cfg.lidar_height_pixel, cfg.lidar_width_pixel
    row_ego = int((0 - cfg.min_y_meter) * ppm); col_ego = int((0 - cfg.min_x_meter) * ppm)
    os.makedirs("outputs/viz_dilation", exist_ok=True)

    cache = {}
    def get_map(t):
        if t not in cache:
            cache[t] = carla.Map(t, open(glob.glob(f"3rd_party/CARLA_0915/**/{t}.xodr", recursive=True)[0]).read())
        return cache[t]

    def heat(x):
        h = np.clip(x, 0, 1)
        return (np.stack([np.clip(h*2,0,1), np.where(h<.5,h*2,1-(h-.5)*2), np.zeros_like(h)], -1)*255).astype(np.uint8)

    # arms with per-point lane width -> ego-frame (col,row) pixel + width_px
    def arms_px(ego_wp, cmap, inv):
        out = []
        for turn in (-1, 0, 1):
            arm = _arm_for_turn(ego_wp, turn)  # world (x,y,z)
            if arm is None:
                continue
            pts, wpx = [], []
            for (wx, wy, wz) in arm:
                v = inv @ np.array([wx, wy, wz, 1.0])
                pts.append((col_ego + v[0]*ppm, row_ego + v[1]*ppm))
            out.append(np.array(pts))
        return out

    for saved, IDX in enumerate([9, 40, 68, 70]):
        ci = vds.valid_indices[IDX]
        base = "/".join(str(cds.images[ci], encoding="utf-8").split("/")[:-2])
        fr = str(cds.images[ci], encoding="utf-8").split("/")[-1].split(".")[0]
        m = common_utils.read_pickle(f"{base}/metas/{fr}.pkl")
        town = m["town"]; inv = np.linalg.inv(np.array(m["ego_matrix"]))
        cmap = get_map(town)
        ego_wp = cmap.get_waypoint(carla.Location(x=m["pos_global"][0], y=m["pos_global"][1], z=m["pos_global"][2]))
        arms = arms_px(ego_wp, cmap, inv)

        # A: thin centerline + gaussian
        A = np.zeros((H, W), np.float32)
        for pts in arms:
            p = pts.astype(np.int32)
            cv2.polylines(A, [p], False, 1.0, 1)
        A = ndimage.gaussian_filter(A, 3.0); A /= (A.max() + 1e-6)
        # B: solid lane-width band (3.5m -> px)
        lane_w_px = int(round(3.5 * ppm))
        B = np.zeros((H, W), np.float32)
        for pts in arms:
            p = pts.astype(np.int32)
            cv2.polylines(B, [p], False, 1.0, thickness=lane_w_px)
        # C: B + gaussian soft edge
        C = ndimage.gaussian_filter(B, 2.0); C /= (C.max() + 1e-6)
        # centerline viz
        CL = _hdmap_bev(str(cds.bev_semantics[ci], encoding="utf-8"), cfg)
        for i, pts in enumerate(arms):
            cv2.polylines(CL, [pts.astype(np.int32)], False, [(255,60,60),(60,255,60),(80,160,255)][i%3], 2)

        hd = _hdmap_bev(str(cds.bev_semantics[ci], encoding="utf-8"), cfg)
        sep = np.full((H, 2, 3), 255, np.uint8)
        combined = np.concatenate([hd, sep, heat(A), sep, heat(B), sep, heat(C), sep, CL], axis=1)
        panel = cv2.rotate(combined, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.putText(panel, f"idx{IDX} {town}: hdmap|A gauss|B lanewidth|C both|centerline (fwdUP)",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        outp = f"outputs/viz_dilation/{saved:02d}_idx{IDX}.png"
        cv2.imwrite(outp, panel[..., ::-1])
        print(f"saved {outp}  {town}  arms={len(arms)}")


if __name__ == "__main__":
    main()
