"""Verify path-only sidecars against frozen B3a feature shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lead.tfv6.future_collision import interpolate_route_by_distance


def verify_shard(feature_path: Path, path_path: Path, samples: int = 128) -> dict:
    with np.load(feature_path, allow_pickle=False) as feature_file:
        keys = feature_file["keys"]
        speed = feature_file["current_speed"].astype(np.float32)
        target = feature_file["raw_target_speed"].astype(np.float32)
        arm = feature_file["selected_arm"]
        source = str(feature_file["source_checkpoint"])
        manifest = str(feature_file["source_manifest"])
        vlm_manifest = str(feature_file["nearest_vlm_manifest"])
        pair_id = (
            str(feature_file["extraction_pair_id"])
            if "extraction_pair_id" in feature_file
            else None
        )
        rows = np.unique(
            np.linspace(0, len(keys) - 1, min(samples, len(keys)), dtype=int)
        )
        velocities = feature_file["candidate_velocity"][rows, 0].astype(np.float32)
        positions = feature_file["candidate_states"][rows, 0, :, :2].astype(np.float32)
    with np.load(path_path, allow_pickle=False) as sidecar:
        sidecar_pair_id = (
            str(sidecar["extraction_pair_id"])
            if "extraction_pair_id" in sidecar
            else None
        )
        if pair_id != sidecar_pair_id:
            raise ValueError(f"same-forward pair ID mismatch: {path_path}")
        if not np.array_equal(keys, sidecar["keys"]):
            raise ValueError(f"keys/order mismatch: {path_path}")
        if str(sidecar["source_checkpoint"]) != source:
            raise ValueError(f"checkpoint mismatch: {path_path}")
        if str(sidecar["source_manifest"]) != manifest:
            raise ValueError(f"manifest mismatch: {path_path}")
        if str(sidecar["nearest_vlm_manifest"]) != vlm_manifest:
            raise ValueError(f"VLM manifest mismatch: {path_path}")
        if not np.array_equal(arm, sidecar["selected_arm"]):
            raise ValueError(f"selected arm mismatch: {path_path}")
        for name, old in (("current_speed", speed), ("raw_target_speed", target)):
            if not np.allclose(sidecar[name], old, atol=0.03):
                raise ValueError(f"{name} mismatch: {path_path}")
        paths = sidecar["selected_path"]
        if paths.ndim != 3 or paths.shape[0] != len(keys) or paths.shape[2] != 2:
            raise ValueError(f"invalid selected Path shape {paths.shape}: {path_path}")
        if not np.isfinite(paths).all():
            raise ValueError(f"nonfinite selected Path: {path_path}")
        errors = []
        interval = 0.25
        for row, velocity, expected in zip(rows, velocities, positions, strict=True):
            distances = np.cumsum(velocity) * interval
            predicted, _ = interpolate_route_by_distance(
                paths[row], distances, extrapolate=True
            )
            errors.extend(np.linalg.norm(predicted - expected, axis=1).tolist())
    error = np.asarray(errors, dtype=np.float32)
    maximum = float(error.max()) if len(error) else 0.0
    if maximum > 0.15:
        raise ValueError(
            f"candidate geometry mismatch: max={maximum:.3f}m in {path_path}"
        )
    return {
        "shard": str(path_path),
        "frames": len(keys),
        "path_points": int(paths.shape[1]),
        "sampled_candidate_point_error_mean_m": float(error.mean()),
        "sampled_candidate_point_error_max_m": maximum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--selected-paths", type=Path, required=True)
    parser.add_argument("--samples-per-shard", type=int, default=128)
    args = parser.parse_args()
    if args.samples_per_shard < 1:
        parser.error("samples-per-shard must be positive")
    feature_paths = sorted(args.features.glob("velocity_features_*.npz"))
    if not feature_paths:
        raise FileNotFoundError(f"no feature shards in {args.features}")
    results = []
    for feature_path in feature_paths:
        sidecar = args.selected_paths / feature_path.name.replace(
            "velocity_features_", "selected_paths_"
        )
        if not sidecar.is_file():
            raise FileNotFoundError(f"missing selected Path shard: {sidecar}")
        result = verify_shard(feature_path, sidecar, args.samples_per_shard)
        results.append(result)
        print(
            f"verified {sidecar.name}: {result['frames']} frames, "
            f"geometry max={result['sampled_candidate_point_error_max_m']:.3f}m"
        )
    print(
        json.dumps(
            {"frames": sum(r["frames"] for r in results), "shards": results}, indent=2
        )
    )


if __name__ == "__main__":
    main()
