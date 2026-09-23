"""Build manifest-aligned future-actor GT cache shards for B2d."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import lzma
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _raw_paths(item: dict, repo_root: Path) -> tuple[Path, Path]:
    image = Path(item["src"])
    if not image.is_absolute():
        image = repo_root / image
    route_dir = image.parent.parent
    return (
        route_dir / "metas" / f"{item['frame']}.pkl",
        route_dir / "bboxes" / f"{item['frame']}.pkl",
    )


def _read_pickle(path: Path):
    with lzma.open(path, "rb") as handle:
        return pickle.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/p4/manifest.jsonl")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--max-actors", type=int, default=90)
    parser.add_argument("--max-actor-distance", type=float, default=80.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent bbox readers/decompressors within each output shard",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least one")

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from lead.data_loader.future_actor_cache import (
        cache_shard_path,
        pack_future_actor_frames,
        parse_future_actor_frame,
    )

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    with manifest_path.open() as handle:
        manifest = [json.loads(line) for line in handle if line.strip()]
    requested = len(manifest) if args.limit is None else min(args.limit, len(manifest))
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def read_frame(item: dict):
        _, bbox_path = _raw_paths(item, repo_root)
        return parse_future_actor_frame(
            _read_pickle(bbox_path),
            max_actors=args.max_actors,
            max_actor_distance_m=args.max_actor_distance,
        )

    built = skipped = 0
    total_dropped = 0
    executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
        if args.workers > 1
        else None
    )
    progress = tqdm(total=requested, desc="future actors", unit="frame")
    try:
        for start in range(0, requested, args.shard_size):
            # Always finish a whole shard. This prevents overlapping partial shards if a
            # later run extends --limit from 5000 to the complete manifest.
            end = min(start + args.shard_size, len(manifest))
            output = cache_shard_path(cache_dir, start, end)
            if output.exists() and not args.overwrite:
                skipped += end - start
                progress.update(end - start)
                continue
            items = manifest[start:end]
            frame_iterator = (
                executor.map(read_frame, items)
                if executor is not None
                else map(read_frame, items)
            )
            frames = []
            keys = []
            for item, frame in zip(items, frame_iterator, strict=True):
                frames.append(frame)
                keys.append(item["key"])
                total_dropped += frame.dropped_actors
                progress.update(1)
            packed = pack_future_actor_frames(frames, keys, max_actors=args.max_actors)
            temporary = output.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **packed)
            os.replace(temporary, output)
            built += end - start
    finally:
        progress.close()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    metadata = {
        "version": 1,
        "manifest": str(manifest_path),
        "manifest_frames": len(manifest),
        "requested_frames": requested,
        "built_frames": built,
        "skipped_frames": skipped,
        "shard_size": args.shard_size,
        "max_actors": args.max_actors,
        "max_actor_distance_m": args.max_actor_distance,
        "workers": args.workers,
        "dropped_actors_in_built_shards": total_dropped,
        "time_grid_s": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        "classes": ["car", "walker", "static", "static_prop_car"],
    }
    with (cache_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
