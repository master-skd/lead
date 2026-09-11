"""Ground-truth future actor parsing and compact, manifest-aligned cache shards.

The raw Bench2Drive boxes store actor futures at CARLA's 20 Hz in the coordinate
frame of the current ego vehicle.  This module samples the same 4 Hz / 2 s grid
as the planning decoder and, importantly, keeps a validity mask instead of using
the constant-velocity tail that the detection auxiliary task uses for training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


FUTURE_STEPS = 8
RAW_STEP_STRIDE = 5
FUTURE_DT_S = 0.25
FUTURE_TIMES_S = np.arange(1, FUTURE_STEPS + 1, dtype=np.float32) * FUTURE_DT_S

ACTOR_CLASS_TO_ID = {
    "car": 1,
    "walker": 2,
    "static": 3,
    "static_prop_car": 4,
}
ACTOR_ID_TO_CLASS = {value: key for key, value in ACTOR_CLASS_TO_ID.items()}


@dataclass(frozen=True)
class FutureActorFrame:
    """Future collision geometry for one frame.

    All extents are CARLA half-extents. Positions and yaws are expressed in the
    current ego frame. Dynamic trajectories can be partially valid at the end.
    """

    positions: np.ndarray  # [A, T, 2]
    yaws: np.ndarray  # [A, T]
    extents: np.ndarray  # [A, 3]
    z: np.ndarray  # [A]
    valid: np.ndarray  # [A, T]
    actor_ids: np.ndarray  # [A]
    class_ids: np.ndarray  # [A]
    ego_extent: np.ndarray  # [3]
    dropped_actors: int = 0

    @property
    def num_actors(self) -> int:
        return int(self.actor_ids.shape[0])


def _finite_vector(value, size: int) -> np.ndarray | None:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size < size or not np.isfinite(array[:size]).all():
        return None
    return array[:size]


def parse_future_actor_frame(
    raw_boxes: list[dict],
    *,
    max_actors: int = 90,
    max_actor_distance_m: float = 80.0,
    max_future_jump_m: float = 10.0,
) -> FutureActorFrame:
    """Parse a raw bbox pickle without inventing future GT after disappearance."""

    ego_extent = np.array([2.45, 0.95, 0.75], dtype=np.float32)
    candidates: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]] = []

    for box in raw_boxes:
        cls = box.get("class")
        extent = _finite_vector(box.get("extent", []), 3)
        current = _finite_vector(box.get("position", []), 3)
        if cls == "ego_car":
            if extent is not None and np.all(extent > 0):
                ego_extent = extent
            continue
        if cls not in ACTOR_CLASS_TO_ID or extent is None or current is None:
            continue
        if np.any(extent <= 0):
            continue

        positions = np.zeros((FUTURE_STEPS, 2), dtype=np.float32)
        yaws = np.zeros(FUTURE_STEPS, dtype=np.float32)
        valid = np.zeros(FUTURE_STEPS, dtype=bool)
        current_yaw = float(box.get("yaw", 0.0))
        if not np.isfinite(current_yaw):
            continue

        if cls in ("static", "static_prop_car"):
            positions[:] = current[:2]
            yaws[:] = current_yaw
            valid[:] = True
        else:
            raw_positions = np.asarray(box.get("future_positions", []), dtype=np.float32)
            raw_yaws = np.asarray(box.get("future_yaws", []), dtype=np.float32).reshape(-1)
            if raw_positions.ndim != 2 or raw_positions.shape[1] < 2:
                raw_positions = np.empty((0, 2), dtype=np.float32)
            last_position = current[:2]
            for step, raw_index in enumerate(range(RAW_STEP_STRIDE, 41, RAW_STEP_STRIDE)):
                if raw_index >= len(raw_positions) or raw_index >= len(raw_yaws):
                    break
                position = raw_positions[raw_index, :2]
                yaw = raw_yaws[raw_index]
                if not np.isfinite(position).all() or not np.isfinite(yaw):
                    break
                # CARLA sometimes teleports disappearing actors far away.
                if np.linalg.norm(position - last_position) > max_future_jump_m:
                    break
                positions[step] = position
                yaws[step] = yaw
                valid[step] = True
                last_position = position

        if not valid.any():
            continue
        min_distance = float(np.linalg.norm(positions[valid], axis=1).min())
        if min_distance > max_actor_distance_m:
            continue
        actor_id = int(box.get("id", -1))
        candidates.append(
            (min_distance, actor_id, positions, yaws, extent, float(current[2]), valid),
        )

    candidates.sort(key=lambda item: item[0])
    dropped = max(0, len(candidates) - max_actors)
    candidates = candidates[:max_actors]
    count = len(candidates)
    if count == 0:
        return FutureActorFrame(
            positions=np.zeros((0, FUTURE_STEPS, 2), dtype=np.float32),
            yaws=np.zeros((0, FUTURE_STEPS), dtype=np.float32),
            extents=np.zeros((0, 3), dtype=np.float32),
            z=np.zeros(0, dtype=np.float32),
            valid=np.zeros((0, FUTURE_STEPS), dtype=bool),
            actor_ids=np.zeros(0, dtype=np.int64),
            class_ids=np.zeros(0, dtype=np.uint8),
            ego_extent=ego_extent,
            dropped_actors=dropped,
        )

    # Recover class IDs by ID. IDs are unique within a raw frame.
    id_to_class = {
        int(box.get("id", -1)): ACTOR_CLASS_TO_ID.get(box.get("class"), 0)
        for box in raw_boxes
    }
    return FutureActorFrame(
        positions=np.stack([item[2] for item in candidates]),
        yaws=np.stack([item[3] for item in candidates]),
        extents=np.stack([item[4] for item in candidates]),
        z=np.asarray([item[5] for item in candidates], dtype=np.float32),
        valid=np.stack([item[6] for item in candidates]),
        actor_ids=np.asarray([item[1] for item in candidates], dtype=np.int64),
        class_ids=np.asarray([id_to_class[item[1]] for item in candidates], dtype=np.uint8),
        ego_extent=ego_extent,
        dropped_actors=dropped,
    )


def pack_future_actor_frames(
    frames: list[FutureActorFrame], keys: list[str], max_actors: int = 90,
) -> dict[str, np.ndarray]:
    """Pad a cache shard into arrays that can be loaded without pickle."""

    if len(frames) != len(keys):
        raise ValueError("frames and keys must have equal length")
    n = len(frames)
    positions = np.zeros((n, max_actors, FUTURE_STEPS, 2), dtype=np.float16)
    yaws = np.zeros((n, max_actors, FUTURE_STEPS), dtype=np.float16)
    extents = np.zeros((n, max_actors, 3), dtype=np.float16)
    z = np.zeros((n, max_actors), dtype=np.float16)
    valid = np.zeros((n, max_actors, FUTURE_STEPS), dtype=bool)
    actor_ids = np.full((n, max_actors), -1, dtype=np.int64)
    class_ids = np.zeros((n, max_actors), dtype=np.uint8)
    ego_extent = np.zeros((n, 3), dtype=np.float16)
    counts = np.zeros(n, dtype=np.uint16)
    dropped = np.zeros(n, dtype=np.uint16)
    for index, frame in enumerate(frames):
        count = min(frame.num_actors, max_actors)
        positions[index, :count] = frame.positions[:count]
        yaws[index, :count] = frame.yaws[:count]
        extents[index, :count] = frame.extents[:count]
        z[index, :count] = frame.z[:count]
        valid[index, :count] = frame.valid[:count]
        actor_ids[index, :count] = frame.actor_ids[:count]
        class_ids[index, :count] = frame.class_ids[:count]
        ego_extent[index] = frame.ego_extent
        counts[index] = count
        dropped[index] = frame.dropped_actors + max(0, frame.num_actors - max_actors)
    return {
        "keys": np.asarray(keys),
        "positions": positions,
        "yaws": yaws,
        "extents": extents,
        "z": z,
        "valid": valid,
        "actor_ids": actor_ids,
        "class_ids": class_ids,
        "ego_extent": ego_extent,
        "counts": counts,
        "dropped": dropped,
        "times_s": FUTURE_TIMES_S,
    }


def unpack_future_actor_frame(shard, index: int) -> FutureActorFrame:
    count = int(shard["counts"][index])
    return FutureActorFrame(
        positions=shard["positions"][index, :count].astype(np.float32),
        yaws=shard["yaws"][index, :count].astype(np.float32),
        extents=shard["extents"][index, :count].astype(np.float32),
        z=shard["z"][index, :count].astype(np.float32),
        valid=shard["valid"][index, :count].astype(bool),
        actor_ids=shard["actor_ids"][index, :count].astype(np.int64),
        class_ids=shard["class_ids"][index, :count].astype(np.uint8),
        ego_extent=shard["ego_extent"][index].astype(np.float32),
        dropped_actors=int(shard["dropped"][index]),
    )


def cache_shard_path(cache_dir: str | Path, start: int, end: int) -> Path:
    return Path(cache_dir) / f"future_actors_{start:06d}_{end:06d}.npz"
