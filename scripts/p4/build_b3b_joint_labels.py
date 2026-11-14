"""Generate GT-privileged labels for same-pass B3b Path x scalar-speed candidates.

The candidate motion uses the existing longitudinal PID proxy and bounded-yaw
path follower. These are training labels, not CARLA dynamics or runtime GT.
"""

from __future__ import annotations

import argparse
import json
import lzma
import pickle
import sqlite3
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lead.data_loader.future_actor_cache import unpack_future_actor_frame
from lead.tfv6.control_aware_velocity_oracle import ACTION_NAMES, rollout_scalar_target, scalar_actions
from lead.tfv6.future_collision import future_collision_label
from scripts.p4.eval_b3b_kinematic_oracle import rollout
from scripts.p4.audit_b3b_pid_local_feasibility import hdmap_path

HEADS = ("collision_free", "ttc", "drivable", "task", "progress", "comfort", "imitation")
STEPS, ACTIONS = 8, len(ACTION_NAMES)


def build_actor_index(cache_dir: Path, index_path: Path) -> None:
    paths = sorted(cache_dir.glob("future_actors_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no future-actor shards under {cache_dir}")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_suffix(".tmp.sqlite")
    if tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    try:
        db.execute("CREATE TABLE frames (key TEXT PRIMARY KEY, shard TEXT NOT NULL, row INTEGER NOT NULL)")
        for path in tqdm(paths, desc="index future actors"):
            with np.load(path, allow_pickle=False) as archive:
                keys = archive["keys"]
            db.executemany("INSERT INTO frames VALUES (?,?,?)",
                           ((str(key), path.name, i) for i, key in enumerate(keys)))
        db.commit()
        db.execute("CREATE TABLE info (cache_dir TEXT, frames INTEGER)")
        db.execute("INSERT INTO info VALUES (?,?)", (str(cache_dir.resolve()),
                    db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]))
        db.commit()
    finally:
        db.close()
    tmp.replace(index_path)
    print(f"indexed {len(paths)} actor shards -> {index_path}")


def load_actors(cache_dir: Path, index_path: Path, keys: np.ndarray) -> dict:
    # The index is atomically finalized and never modified. On Ceph, ordinary
    # SQLite read locks can block for minutes in fcntl; immutable skips them.
    db = sqlite3.connect(f"file:{index_path.resolve()}?mode=ro&immutable=1", uri=True)
    try:
        info = db.execute("SELECT cache_dir FROM info").fetchone()
        if info is None or info[0] != str(cache_dir.resolve()):
            raise ValueError("actor index belongs to another cache directory")
        locations = defaultdict(list)
        for key in map(str, keys):
            hit = db.execute("SELECT shard,row FROM frames WHERE key=?", (key,)).fetchone()
            if hit is None:
                raise KeyError(f"future actors missing for {key}")
            locations[hit[0]].append((key, int(hit[1])))
    finally:
        db.close()
    result = {}
    for name, selected in locations.items():
        with np.load(cache_dir / name, allow_pickle=False) as archive:
            data = {field: archive[field] for field in archive.files}
        for key, row in selected:
            result[key] = unpack_future_actor_frame(data, row)
    return result


def road_fraction_fast(hdmap: np.ndarray, xy: np.ndarray, yaw: np.ndarray) -> float:
    """GT road fraction at 8 poses, center and +-0.8m sides (not swept)."""
    normal = np.stack((-np.sin(yaw), np.cos(yaw)), axis=-1)
    points = xy[:, None, :] + np.array([-0.8, 0, 0.8], dtype=np.float32)[None, :, None] * normal[:, None, :]
    center = hdmap.shape[0] / 2
    col = np.rint(center + 2 * points[..., 0]).astype(int)
    row = np.rint(center + 2 * points[..., 1]).astype(int)
    inside = (row >= 0) & (row < hdmap.shape[0]) & (col >= 0) & (col < hdmap.shape[1])
    road = np.zeros_like(inside)
    road[inside] = np.isin(hdmap[row[inside], col[inside]], (1, 3, 4))
    return float(road.mean())


def make_frame(routes, route_valid, winner, current_speed, raw_target, nav_route,
               expert_future, actors, hdmap, margin=0.2):
    paths = len(routes)
    candidates = paths * ACTIONS
    states = np.zeros((candidates, STEPS, 6), dtype=np.float16)
    valid = np.zeros(candidates, dtype=bool)
    labels = {name: np.zeros(candidates, dtype=np.float16) for name in HEADS}
    target_speed = np.zeros(candidates, dtype=np.float16)
    tracking_error = np.zeros(candidates, dtype=np.float16)
    targets, action_valid = scalar_actions(raw_target)
    action_rollouts = [rollout_scalar_target(max(0.0, current_speed), float(target))
                       for target in targets]
    raw_progress = max(action_rollouts[0].progress_m, 0.25)
    expert_end = np.asarray(nav_route[-1], dtype=np.float32)
    expert_heading = np.arctan2(expert_end[1], expert_end[0])
    for path_index in range(paths):
        if not route_valid[path_index]:
            continue
        route = np.asarray(routes[path_index], dtype=np.float32)
        end = route[-1]
        end_heading = np.arctan2(end[1], end[0])
        angle_error = abs(np.arctan2(np.sin(end_heading - expert_heading),
                                    np.cos(end_heading - expert_heading)))
        task = float(np.exp(-0.5 * (angle_error / np.deg2rad(35)) ** 2))
        # Path imitation is route geometry; the velocity part enters through
        # time-sampled ego positions against the expert's corresponding points.
        for action_index, action in enumerate(action_rollouts):
            flat = path_index * ACTIONS + action_index
            if not action_valid[action_index]:
                continue
            distances = action.xy_distance_m
            speeds = np.diff(np.r_[0., distances]) / .25
            xy, yaw, error = rollout(route, speeds)
            acc = np.diff(np.r_[max(0.0, current_speed), action.speed_mps]) / .25
            states[flat, :, 0:2] = xy
            states[flat, :, 2] = np.sin(yaw)
            states[flat, :, 3] = np.cos(yaw)
            states[flat, :, 4] = action.speed_mps
            states[flat, :, 5] = acc
            valid[flat] = True
            target_speed[flat] = action.target_mps
            tracking_error[flat] = error
            collision = future_collision_label(xy, yaw, actors, safety_margin_m=margin,
                                                include_class_ids=(1, 2))
            labels["collision_free"][flat] = float(not collision.collision)
            labels["ttc"][flat] = (min(collision.ttc_s / 2, 1.0)
                                    if collision.collision else 1.0)
            labels["drivable"][flat] = road_fraction_fast(hdmap, xy, yaw)
            labels["task"][flat] = task
            labels["progress"][flat] = min(action.progress_m / raw_progress, 1.0)
            labels["comfort"][flat] = action.comfort * np.exp(-max(0., error - .5))
            # Full spatiotemporal imitation: GT ego positions at matching 4Hz
            # instants. Cached `expert_route` is actually a NAV route, not GT ego.
            ade = float(np.linalg.norm(xy - expert_future, axis=-1).mean())
            labels["imitation"][flat] = np.exp(-ade / 2.0)
    baseline_index = int(winner) * ACTIONS
    if not 0 <= baseline_index < candidates or not valid[baseline_index]:
        raise ValueError("same-pass confidence winner is not a valid baseline candidate")
    return states, valid, labels, target_speed, tracking_error, baseline_index


def process(scene_path: Path, actor_dir: Path, actor_index: Path, hdmap_root: Path,
            output: Path, limit: int = 0, margin: float = 0.2) -> None:
    cv2.setNumThreads(0)
    with np.load(scene_path, allow_pickle=False) as archive:
        required = ("keys", "routes", "route_valid", "winner", "current_speed",
                    "raw_target_speed", "expert_route", "same_forward", "source_checkpoint")
        if any(name not in archive.files for name in required):
            raise ValueError("not a B3b same-forward scene shard")
        if not bool(archive["same_forward"]):
            raise ValueError("scene shard is not same-forward")
        n = min(len(archive["keys"]), limit) if limit else len(archive["keys"])
        source = {key: archive[key][:n] for key in required if key not in
                  ("same_forward", "source_checkpoint")}
        checkpoint = str(archive["source_checkpoint"])
    if len(set(map(str, source["keys"]))) != n:
        raise ValueError("duplicate keys in scene shard")
    actors = load_actors(actor_dir, actor_index, source["keys"])
    candidates = source["routes"].shape[1] * ACTIONS
    states = np.zeros((n, candidates, STEPS, 6), dtype=np.float16)
    valid = np.zeros((n, candidates), dtype=bool)
    labels = {name: np.zeros((n, candidates), dtype=np.float16) for name in HEADS}
    targets = np.zeros((n, candidates), dtype=np.float16)
    errors = np.zeros((n, candidates), dtype=np.float16)
    baseline = np.zeros(n, dtype=np.int16)
    for i in tqdm(range(n), desc=f"GT joint labels {scene_path.stem}"):
        key = str(source["keys"][i])
        map_path = hdmap_path(hdmap_root, key)
        hdmap = cv2.imread(str(map_path), cv2.IMREAD_UNCHANGED)
        if hdmap is None or hdmap.ndim != 2:
            raise FileNotFoundError(f"raw GT HD-map missing or invalid: {map_path}")
        meta_path = map_path.parent.parent / "metas" / (map_path.stem + ".pkl")
        with lzma.open(meta_path, "rb") as stream:
            meta = pickle.load(stream)
        future = np.asarray(meta["future_positions"], dtype=np.float32)
        if future.ndim != 2 or future.shape[0] <= 40 or future.shape[1] < 2:
            raise ValueError(f"incomplete GT ego future: {meta_path}")
        expert_future = future[np.arange(5, 41, 5), :2]
        result = make_frame(source["routes"][i], source["route_valid"][i],
                            int(source["winner"][i]), float(source["current_speed"][i]),
                            float(source["raw_target_speed"][i]), source["expert_route"][i],
                            expert_future,
                            actors[key], hdmap, margin)
        states[i], valid[i], frame_labels, targets[i], errors[i], baseline[i] = result
        for name in HEADS:
            labels[name][i] = frame_labels[name]
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.stem + ".tmp.npz")
    np.savez_compressed(tmp, keys=source["keys"], candidate_states=states,
                        candidate_valid=valid, target_speed=targets,
                        tracking_error=errors, baseline_index=baseline,
                        **{f"label_{name}": value for name, value in labels.items()},
                        source_scene=str(scene_path.resolve()), source_checkpoint=checkpoint,
                        schema_version=1, safety_margin_m=margin,
                        action_names=np.asarray(ACTION_NAMES))
    with np.load(tmp, allow_pickle=False) as check:
        if len(check["keys"]) != n or not np.all(check["candidate_valid"][np.arange(n), baseline]):
            raise ValueError("written label shard failed baseline/count validation")
    tmp.replace(output)
    unsafe = valid & (labels["collision_free"] < .5)
    print(json.dumps({"output": str(output), "frames": n,
                      "candidates_per_frame": float(valid.sum() / n),
                      "raw_collision_rate": float(unsafe[np.arange(n), baseline].mean()),
                      "candidate_collision_rate": float(unsafe.sum() / max(valid.sum(), 1))}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path)
    p.add_argument("--actor-dir", type=Path, required=True)
    p.add_argument("--actor-index", type=Path, required=True)
    p.add_argument("--hdmap-root", type=Path, default=Path("data/carla_leaderboard2/data"))
    p.add_argument("--output", type=Path)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--margin", type=float, default=.2)
    p.add_argument("--build-index", action="store_true")
    p.add_argument("--all", action="store_true", help="process every scene shard with CPU workers")
    p.add_argument("--scene-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.build_index:
        build_actor_index(args.actor_dir, args.actor_index)
    elif args.all:
        if args.scene_dir is None or args.output_dir is None or args.workers < 1:
            p.error("--all requires --scene-dir, --output-dir and positive --workers")
        if not args.actor_index.exists():
            build_actor_index(args.actor_dir, args.actor_index)
        scenes = sorted(args.scene_dir.glob("joint_scene_*.npz"))
        if not scenes:
            raise FileNotFoundError(args.scene_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pending = []
        for scene in scenes:
            output = args.output_dir / scene.name.replace("joint_scene_", "joint_labels_")
            if output.exists():
                with np.load(scene, allow_pickle=False) as s, np.load(output, allow_pickle=False) as l:
                    if not np.array_equal(s["keys"], l["keys"]) or str(s["source_checkpoint"]) != str(l["source_checkpoint"]):
                        raise ValueError(f"existing output mismatch: {output}")
                continue
            pending.append((scene, output))
        print(f"scene shards={len(scenes)} pending={len(pending)} workers={args.workers}", flush=True)

        def run_one(item):
            scene, output = item
            log = args.output_dir / (output.stem + ".log")
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--scene", str(scene), "--output", str(output),
                       "--actor-dir", str(args.actor_dir),
                       "--actor-index", str(args.actor_index),
                       "--hdmap-root", str(args.hdmap_root),
                       "--margin", str(args.margin)]
            with log.open("w") as stream:
                done = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            return output, log, done.returncode

        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, item) for item in pending]
            for future in as_completed(futures):
                output, log, code = future.result()
                print(f"{'done' if code == 0 else 'FAILED'} {output} log={log}", flush=True)
                if code:
                    failures.append(str(log))
        if failures:
            raise RuntimeError(f"{len(failures)} label shards failed: {failures[:5]}")
        seen = set()
        frames = 0
        for scene in scenes:
            output = args.output_dir / scene.name.replace("joint_scene_", "joint_labels_")
            if not output.exists():
                raise FileNotFoundError(output)
            with np.load(scene, allow_pickle=False) as s, np.load(output, allow_pickle=False) as l:
                keys = l["keys"]
                valid = l["candidate_valid"]
                baseline = l["baseline_index"]
                if not np.array_equal(s["keys"], keys) or not np.all(valid[np.arange(len(keys)), baseline]):
                    raise ValueError(f"invalid label shard: {output}")
                if str(l["source_scene"]) != str(scene.resolve()) or str(l["source_checkpoint"]) != str(s["source_checkpoint"]):
                    raise ValueError(f"label provenance mismatch: {output}")
                for name in HEADS:
                    label = l[f"label_{name}"]
                    if not np.isfinite(label).all() or np.any(label[valid] < 0) or np.any(label[valid] > 1):
                        raise ValueError(f"nonfinite/out-of-range {name}: {output}")
                new_keys = set(map(str, keys))
                if new_keys & seen:
                    raise ValueError(f"duplicate frame keys in {output}")
                seen.update(new_keys)
                frames += len(keys)
        print(json.dumps({"verified_shards": len(scenes), "frames": frames,
                          "unique_keys": len(seen)}), flush=True)
    else:
        if args.scene is None or args.output is None:
            p.error("--scene and --output required unless --build-index")
        process(args.scene, args.actor_dir, args.actor_index, args.hdmap_root,
                args.output, args.limit, args.margin)


if __name__ == "__main__":
    main()
