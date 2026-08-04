"""P5 B2: lane-graph arm extraction using scenario_runner's junction logic
(generate_target_waypoint_list / choose_at_junction), reimplemented offline so it works
without CarlaDataProvider/server. For each frame, expand LEFT/STRAIGHT/RIGHT arms from ego
through the next junction (capped at HORIZON), transform to ego-BEV, overlay on hdmap.

Fixes the earlier .next()-recursion that missed branches when ego was already inside a
junction: choose_at_junction uses the cross-product of ego heading vs each successor to
classify left/straight/right, so all exits are reachable regardless of ego position.
"""
from __future__ import annotations
import json, os, glob, math
import numpy as np

HORIZON = 30.0
STEP = 2.0


def _choose_at_junction(current_wp, next_choices, direction):
    """scenario_helper.choose_at_junction, inlined. direction: -1 left, 0 straight, 1 right."""
    import carla
    ct = current_wp.transform
    cl = ct.location
    proj = cl + carla.Location(x=math.cos(math.radians(ct.rotation.yaw)),
                               y=math.sin(math.radians(ct.rotation.yaw)))
    cur_vec = np.array([proj.x - cl.x, proj.y - cl.y, 0.0])
    crosses, cmap = [], {}
    for wp in next_choices:
        w = wp.next(10)[0]
        sel = np.array([w.transform.location.x - cl.x, w.transform.location.y - cl.y, 0.0])
        cz = float(np.cross(cur_vec, sel)[2])
        crosses.append(cz); cmap[cz] = wp
    if direction > 0:
        key = max(crosses)
    elif direction < 0:
        key = min(crosses)
    else:
        key = min(crosses, key=abs)
    return cmap[key]


def _arm_for_turn(ego_wp, turn, horizon=HORIZON, step=STEP):
    """Follow waypoints from ego through the next junction, taking `turn` (-1/0/1).
    Returns a list of (x,y,z) world points, capped at horizon. None if degenerate."""
    plan = [ego_wp]
    reached_junction = False
    dist = 0.0
    threshold = math.radians(0.1)
    while dist < horizon:
        nxts = plan[-1].next(step)
        if not nxts:
            break
        if len(nxts) > 1:
            reached_junction = True
            nxt = _choose_at_junction(plan[-1], nxts, turn)
        else:
            nxt = nxts[0]
        dist += nxt.transform.location.distance(plan[-1].transform.location)
        plan.append(nxt)
        # turn arms: stop once the turn has straightened out past the junction
        if turn != 0 and reached_junction and len(plan) >= 3:
            v1 = np.array([plan[-1].transform.location.x - plan[-2].transform.location.x,
                           plan[-1].transform.location.y - plan[-2].transform.location.y])
            v2 = np.array([plan[-2].transform.location.x - plan[-3].transform.location.x,
                           plan[-2].transform.location.y - plan[-3].transform.location.y])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-3 and n2 > 1e-3:
                ang = math.acos(np.clip(np.dot(v1, v2)/(n1*n2), -1, 1))
                if ang < threshold and dist > 12:
                    break
    if len(plan) < 2:
        return None
    return [(w.transform.location.x, w.transform.location.y, w.transform.location.z) for w in plan]


def main():
    import carla, cv2
    from lead.common import common_utils
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.training.config_training import TrainingConfig
    import sys; sys.path.insert(0, "scripts/p4")
    from viz_b2_arms import _hdmap_bev

    cfg = TrainingConfig(json.load(open("outputs/local_training/p5_stepB2_v2/config.json")),
                         raise_error_on_missing_key=False)
    cds = CARLAData(root=cfg.carla_data, config=cfg)
    vds = VLMIntentDataset(cds, vlm_cache_dir=cfg.vlm_cache_dir, manifest_path=cfg.vlm_manifest)
    ppm = cfg.pixels_per_meter
    H, W = cfg.lidar_height_pixel, cfg.lidar_width_pixel
    row_ego = int((0 - cfg.min_y_meter) * ppm); col_ego = int((0 - cfg.min_x_meter) * ppm)
    os.makedirs("outputs/viz_lanegraph_v2", exist_ok=True)

    cache = {}
    def get_map(town):
        if town not in cache:
            cands = glob.glob(f"3rd_party/CARLA_0915/**/{town}.xodr", recursive=True)
            cache[town] = carla.Map(town, open(cands[0]).read())
        return cache[town]

    COLS = [(255,60,60),(60,255,60),(80,160,255)]
    NAMES = {-1: "L", 0: "S", 1: "R"}
    for saved, IDX in enumerate([9, 26, 40, 68, 70]):
        ci = vds.valid_indices[IDX]
        base = "/".join(str(cds.images[ci], encoding="utf-8").split("/")[:-2])
        fr = str(cds.images[ci], encoding="utf-8").split("/")[-1].split(".")[0]
        m = common_utils.read_pickle(f"{base}/metas/{fr}.pkl")
        town, pos, ego_mat = m["town"], m["pos_global"], np.array(m["ego_matrix"])
        cmap = get_map(town)
        ego_wp = cmap.get_waypoint(carla.Location(x=pos[0], y=pos[1], z=pos[2]))
        inv = np.linalg.inv(ego_mat)
        canvas = _hdmap_bev(str(cds.bev_semantics[ci], encoding="utf-8"), cfg)
        cv2.circle(canvas, (col_ego, row_ego), 3, (255,255,255), -1)
        narms = 0
        for turn in (-1, 0, 1):
            arm = _arm_for_turn(ego_wp, turn)
            if arm is None:
                continue
            narms += 1
            pts = []
            for (wx, wy, wz) in arm:
                v = inv @ np.array([wx, wy, wz, 1.0])
                pts.append([col_ego + v[0]*ppm, row_ego + v[1]*ppm])
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, COLS[turn+1], 2)
        canvas = cv2.rotate(canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.putText(canvas, f"idx{IDX} {town} L/S/R arms={narms} (fwdUP)", (5,15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        outp = f"outputs/viz_lanegraph_v2/{saved:02d}_idx{IDX}.png"
        cv2.imwrite(outp, canvas[..., ::-1])
        print(f"saved {outp}  {town}  arms={narms}")


if __name__ == "__main__":
    main()
