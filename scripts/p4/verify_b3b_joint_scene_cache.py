"""Verify B3b scene shards, source identity and split-level frame coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.p4.extract_b3b_joint_scene import verify_shard


def verify(cache_dir: Path, ckpt_dir: Path, manifest: Path, nearest: Path,
           start: int, end: int, chunk_size: int,
           offsets: tuple[float, ...] = (-1.5, -0.75, 0.75, 1.5)) -> dict:
    if start < 0 or end <= start or chunk_size < 1:
        raise ValueError("invalid interval or chunk size")
    checkpoint = ckpt_dir / "model_0019.pth"
    with nearest.open() as handle:
        cache_routes = {(entry["scenario"], entry["route"]) for entry in
                        (json.loads(line) for line in handle if line.strip())}
    with manifest.open() as handle:
        expected = {entry["key"] for entry in
                    (json.loads(line) for line in handle if line.strip())
                    if (entry["scenario"], entry["route"]) in cache_routes}
    seen: set[str] = set()
    count = 0
    shards = 0
    for begin in range(start, end, chunk_size):
        stop = min(begin + chunk_size, end)
        path = cache_dir / f"joint_scene_{begin:06d}_{stop:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = verify_shard(
            path, checkpoint, manifest, nearest, begin, stop, offsets
        )
        with np.load(path, allow_pickle=False) as archive:
            keys = set(map(str, archive["keys"]))
        if seen & keys:
            raise ValueError(f"duplicate keys across shards: {path}")
        if not keys <= expected:
            raise ValueError(f"unexpected frame keys in {path}")
        seen.update(keys)
        count += rows
        shards += 1
    if count != end - start:
        raise ValueError(f"frame count {count} != requested {end-start}")
    if start == 0 and end == len(expected) and seen != expected:
        raise ValueError(f"full split coverage differs: missing {len(expected - seen)}")
    return {"frames": count, "unique_keys": len(seen), "shards": shards,
            "full_split": start == 0 and end == len(expected)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nearest-vlm-manifest", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    args = parser.parse_args()
    result = verify(
        args.cache_dir, args.ckpt_dir, args.manifest,
        args.nearest_vlm_manifest, args.start, args.end, args.chunk_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
