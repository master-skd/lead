"""P5 B2: precompute lane-graph intent labels + anchors for all frames (offline, carla.Map
from .xodr, no server). Replaces the mushy flood-fill blob with a clean multi-arm corridor.

Per frame:
  - locate ego on the lane graph (get_waypoint from meta pos_global)
  - expand L/S/R arms through the next junction (choose_at_junction cross-product logic)
  - intent label: each arm's centerline dilated to lane width (~3.5m) + gaussian soft edge
    (style C), unioned into a single-channel (1,H,W) soft corridor field
  - anchor: per-arm (valid, angle, reach, tip_row, tip_col) from the arm endpoint
Both saved under <out>/<scenario>/<route>/<frame>.npy. Grouped by town so each map loads once.

Run (lead env), sharded:
  for i in 0..7: CUDA_VISIBLE_DEVICES="" python scripts/p4/precompute_lanegraph_intent.py \
     --out-label data/p6/lanegraph_label --out-anchor data/p6/lanegraph_anchor \
     --num-shards 8 --shard $i &
"""
from __future__ import annotations
import argparse, glob, json, math, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/p4/manifest.jsonl")
    ap.add_argument("--out-label", default="data/p6/lanegraph_label")
    ap.add_argument("--out-anchor", default="data/p6/lanegraph_anchor")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import carla, cv2
    from scipy import ndimage
    import sys
    sys.path.insert(0, "scripts/p4")
    from check_lanegraph_arms import _arm_for_turn
    from lead.common import common_utils
    from lead.training.config_training import TrainingConfig

    cfg = TrainingConfig({"use_multimodal_intent": True}, raise_error_on_missing_key=False)
    ppm = cfg.pixels_per_meter
    H, W = cfg.lidar_height_pixel, cfg.lidar_width_pixel
    row_ego = int((0 - cfg.min_y_meter) * ppm)
    col_ego = int((0 - cfg.min_x_meter) * ppm)
    lane_w_px = int(round(3.5 * ppm))
    K_MAX = 6

    entries = [json.loads(l) for l in open(args.manifest)]
    entries = [e for i, e in enumerate(entries) if i % args.num_shards == args.shard]
    if args.limit:
        entries = entries[: args.limit]
    # group by town (route prefix) so each carla.Map loads once
    entries.sort(key=lambda e: e["route"].split("_")[0])
    print(f"[shard {args.shard}/{args.num_shards}] frames: {len(entries)}", flush=True)

    map_cache = {}
    def get_map(town):
        if town not in map_cache:
            hits = glob.glob(f"3rd_party/CARLA_0915/**/{town}.xodr", recursive=True)
            map_cache.clear()  # keep only 1 map in memory (big towns)
            map_cache[town] = carla.Map(town, open(hits[0]).read())
        return map_cache[town]

    done = skipped = bad = 0
    for e in entries:
        sc, rt, fr = e["scenario"], e["route"], e["frame"]
        lab_p = os.path.join(args.out_label, sc, rt, fr + ".npy")
        anc_p = os.path.join(args.out_anchor, sc, rt, fr + ".npy")
        if os.path.exists(lab_p) and os.path.exists(anc_p):
            skipped += 1; continue
        meta_p = os.path.join(os.path.dirname(os.path.dirname(e["src"])), "metas", fr + ".pkl")
        try:
            m = common_utils.read_pickle(meta_p)
            town = m["town"]; pos = m["pos_global"]; inv = np.linalg.inv(np.array(m["ego_matrix"]))
            cmap = get_map(town)
            ego_wp = cmap.get_waypoint(carla.Location(x=pos[0], y=pos[1], z=pos[2]))
        except Exception as exc:  # noqa: BLE001
            bad += 1
            if bad <= 5:
                print(f"  [bad] {sc}/{rt}/{fr}: {exc}", flush=True)
            continue

        band = np.zeros((H, W), np.float32)
        anchors = np.zeros((K_MAX, 5), np.float32)  # [valid, angle, reach, tip_row, tip_col]
        ai = 0
        seen_ang = []  # dedup arms whose direction nearly coincides (curve -> 1 arm, not 3)
        MERGE_DEG = 15.0
        for turn in (-1, 0, 1):
            arm = _arm_for_turn(ego_wp, turn)
            if arm is None:
                continue
            pts = []
            for (wx, wy, wz) in arm:
                v = inv @ np.array([wx, wy, wz, 1.0])
                pts.append([col_ego + v[0] * ppm, row_ego + v[1] * ppm])
            pts = np.array(pts, np.int32)
            cv2.polylines(band, [pts], False, 1.0, thickness=lane_w_px)  # style B: lane width
            tip = pts[-1]
            dr, dc = tip[1] - row_ego, tip[0] - col_ego
            ang = math.atan2(dr, dc)
            # dedup: skip if this arm's angle ~ an already-recorded arm (same physical branch)
            if any(abs(math.degrees(math.atan2(math.sin(ang - a), math.cos(ang - a)))) < MERGE_DEG
                   for a in seen_ang):
                continue
            seen_ang.append(ang)
            if ai < K_MAX:
                anchors[ai] = [1.0, ang, math.hypot(dr, dc) / ppm, tip[1], tip[0]]
                ai += 1
        label = ndimage.gaussian_filter(band, 2.0)  # style C: + gaussian soft edge
        if label.max() > 0:
            label /= label.max()

        os.makedirs(os.path.dirname(lab_p), exist_ok=True)
        os.makedirs(os.path.dirname(anc_p), exist_ok=True)
        np.save(lab_p, label[None].astype(np.float16))          # (1,H,W)
        np.save(anc_p, anchors.astype(np.float16))              # (K_MAX,5)
        done += 1
        if done <= 3 or done % 2000 == 0:
            print(f"[shard {args.shard}] {done} {sc}/{fr} arms={ai} town={town}", flush=True)

    print(f"[shard {args.shard}] finished: done={done} skipped={skipped} bad={bad}", flush=True)


if __name__ == "__main__":
    main()
