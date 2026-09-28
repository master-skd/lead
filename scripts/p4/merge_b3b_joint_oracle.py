"""Merge contiguous B3b oracle shards and recompute global frame-level metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.p4.eval_b3b_joint_oracle import summarize


def summarize_arrays(arrays: dict[str, np.ndarray], report: dict) -> tuple[dict, dict]:
    n = len(arrays["key"])
    winner = arrays["winner"]
    raw_collision = arrays["raw_collision"]
    raw_progress = arrays["raw_progress"]
    scopes = {"all": np.ones(n, dtype=bool), "multi": arrays["multi"]}
    results = {}
    for policy in ("reachable", "conservative"):
        fixed = arrays[f"{policy}_fixed_rescue"]
        joint = arrays[f"{policy}_joint_rescue"]
        joint_arm = arrays[f"{policy}_joint_arm"]
        joint_mode = arrays[f"{policy}_joint_mode"]
        expanded = arrays[f"{policy}_expanded_rescue"]
        expanded_arm = arrays[f"{policy}_expanded_arm"]
        expanded_mode = arrays[f"{policy}_expanded_mode"]
        results[policy] = {}
        for name, scope in scopes.items():
            joint_progress = np.where(
                joint, arrays["candidate_progress"][np.arange(n), joint_mode], raw_progress
            )
            results[policy][name] = summarize(
                raw_collision=raw_collision, fixed_rescue=fixed, joint_rescue=joint,
                joint_arm=joint_arm, winner_arm=winner, raw_progress=raw_progress,
                joint_progress=joint_progress, scope=scope,
            )
            if report["local_offsets_m"]:
                expanded_progress = np.where(
                    expanded,
                    arrays["candidate_progress"][np.arange(n), expanded_mode],
                    raw_progress,
                )
                result = summarize(
                    raw_collision=raw_collision, fixed_rescue=fixed,
                    joint_rescue=expanded, joint_arm=expanded_arm, winner_arm=winner,
                    raw_progress=raw_progress, joint_progress=expanded_progress, scope=scope,
                )
                result["incremental_local_rescue_count"] = int((
                    raw_collision & scope & expanded & ~joint
                ).sum())
                result["local_path_selected_on_rescue_count"] = int((
                    raw_collision & scope & expanded
                    & (expanded_arm >= report["original_path_count"])
                ).sum())
                results[policy][f"expanded_{name}"] = result
    corridor = arrays["corridor_cost"]
    direction = arrays["direction_ok"]
    valid = arrays["route_valid"]
    diagnostics = {
        "winner_direction_proxy_pass_rate": float(direction[np.arange(n), winner].mean()),
        "winner_corridor_pass_rate": float((
            corridor[np.arange(n), winner] <= report["policy"]["max_off_corridor_cost"]
        ).mean()),
        "mean_valid_arms": float(valid.sum(axis=1).mean()),
        "mean_original_valid_arms": float(
            valid[:, :report["original_path_count"]].sum(axis=1).mean()
        ),
        "mean_direction_and_corridor_eligible_arms": float((
            valid & direction & (corridor <= report["policy"]["max_off_corridor_cost"])
        ).sum(axis=1).mean()),
        "mean_reachable_vocabulary_profiles_including_raw": float(
            arrays["candidate_valid"].sum(axis=1).mean()
        ),
        "raw_route_overflow_rate": float((
            raw_progress > arrays["route_length"][np.arange(n), winner] + 1e-3
        ).mean()),
    }
    return results, diagnostics


def merge(shard_paths: list[Path], output: Path) -> dict:
    if not shard_paths:
        raise ValueError("no shards supplied")
    loaded = []
    for path in shard_paths:
        with path.open() as handle:
            report = json.load(handle)
        frame_path = Path(report["frame_records"])
        if not frame_path.is_absolute():
            frame_path = path.parent / frame_path
        with np.load(frame_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if report["end"] - report["start"] != report["n_frames"]:
            raise ValueError(f"shard interval/frame count mismatch: {path}")
        if len(arrays["key"]) != report["n_frames"]:
            raise ValueError(f"shard NPZ/frame count mismatch: {path}")
        loaded.append((path, report, arrays))
    loaded.sort(key=lambda item: item[1]["start"])
    base = loaded[0][1]
    static = (
        "candidate_shape", "original_path_count", "local_offsets_m", "variant_ramp_m",
        "checkpoint", "manifest", "vocabulary", "future_cache_dir",
        "selection_privilege", "corridor_source", "policy",
    )
    keys = set(loaded[0][2])
    end = base["start"]
    for path, report, arrays in loaded:
        if report["start"] != end:
            raise ValueError(f"gap or overlap before {path}: expected {end}, got {report['start']}")
        end = report["end"]
        if any(report[field] != base[field] for field in static):
            raise ValueError(f"incompatible shard metadata: {path}")
        if set(arrays) != keys:
            raise ValueError(f"incompatible NPZ fields: {path}")
        for key in keys:
            if arrays[key].shape[0] != report["n_frames"] or (
                arrays[key].shape[1:] != loaded[0][2][key].shape[1:]
            ):
                raise ValueError(f"incompatible NPZ shape for {key}: {path}")
    arrays = {key: np.concatenate([item[2][key] for item in loaded]) for key in keys}
    if len(np.unique(arrays["key"])) != len(arrays["key"]):
        raise ValueError("duplicate frame keys across shards")
    results, diagnostics = summarize_arrays(arrays, base)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_path = output.with_suffix(".frames.npz")
    np.savez_compressed(frame_path, **arrays)
    merged = dict(base)
    merged.update(
        start=base["start"], end=end, n_frames=len(arrays["key"]),
        diagnostics=diagnostics, results=results,
        frame_records=str(frame_path.resolve()),
        shards=[str(item[0].resolve()) for item in loaded],
    )
    with output.open("w") as handle:
        json.dump(merged, handle, indent=2)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    result = merge(args.shards, args.out)
    print(f"merged {len(args.shards)} shards, {result['n_frames']} frames: {args.out}")


if __name__ == "__main__":
    main()
