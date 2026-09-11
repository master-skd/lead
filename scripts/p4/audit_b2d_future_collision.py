"""Audit time-aligned OBB/TTC labels before training the B2d risk head."""

from __future__ import annotations

import argparse
import json
import lzma
import pickle
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _read_pickle(path: Path):
    with lzma.open(path, "rb") as handle:
        return pickle.load(handle)


def _paths(item: dict, repo_root: Path) -> tuple[Path, Path]:
    image = Path(item["src"])
    if not image.is_absolute():
        image = repo_root / image
    route = image.parent.parent
    return route / "metas" / f"{item['frame']}.pkl", route / "bboxes" / f"{item['frame']}.pkl"


def _binary_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    pred, target = pred.astype(bool), target.astype(bool)
    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    fn = int((~pred & target).sum())
    tn = int((~pred & ~target).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/p4/manifest.jsonl")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--safety-margin", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from lead.data_loader.future_actor_cache import (
        ACTOR_ID_TO_CLASS,
        FUTURE_TIMES_S,
        unpack_future_actor_frame,
    )
    from lead.tfv6.future_collision import (
        future_collision_label,
        route_future_collision_label,
    )

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    with manifest_path.open() as handle:
        manifest = [json.loads(line) for line in handle if line.strip()]
    limit = min(args.limit, len(manifest))

    cache_by_key = {}
    for shard_path in sorted(Path(args.cache_dir).glob("future_actors_*.npz")):
        shard = np.load(shard_path, allow_pickle=False)
        for local_index, key in enumerate(shard["keys"]):
            key = str(key)
            if key in cache_by_key:
                raise RuntimeError(f"duplicate cache key {key} in {shard_path}")
            cache_by_key[key] = (shard_path, local_index)
    missing = [item["key"] for item in manifest[:limit] if item["key"] not in cache_by_key]
    if missing:
        raise FileNotFoundError(
            f"future actor cache misses {len(missing)} requested frames; first={missing[0]}",
        )

    meta_paths = [_paths(item, repo_root)[0] for item in manifest[:limit]]
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        metas = list(
            tqdm(
                executor.map(_read_pickle, meta_paths),
                total=limit,
                desc="load B2d metas",
            ),
        )

    open_shards = {}
    records = []
    class_counts = Counter()
    valid_step_counts = np.zeros(len(FUTURE_TIMES_S), dtype=np.int64)
    scenario_stats = defaultdict(lambda: Counter(frames=0, nominal_collision=0, brake=0, hazard=0))
    dropped_total = 0
    for item, meta in tqdm(
        zip(manifest[:limit], metas, strict=True),
        total=limit,
        desc="B2d future OBB/TTC audit",
    ):
        shard_path, local_index = cache_by_key[item["key"]]
        shard = open_shards.setdefault(shard_path, np.load(shard_path, allow_pickle=False))
        actors = unpack_future_actor_frame(shard, local_index)
        dropped_total += actors.dropped_actors
        valid_step_counts += actors.valid.sum(axis=0)

        for class_id in actors.class_ids:
            class_counts[ACTOR_ID_TO_CLASS[int(class_id)]] += 1

        route = np.asarray(meta["route"], dtype=np.float32)
        current_speed = max(0.0, float(meta.get("speed", 0.0)))
        controlled_target = max(0.0, float(meta.get("target_speed", 0.0)))
        nominal_target = max(
            controlled_target,
            max(0.0, float(meta.get("second_highest_speed", controlled_target))),
        )
        nominal = route_future_collision_label(
            route, current_speed, nominal_target, actors, safety_margin_m=args.safety_margin,
        )
        nominal_dynamic = route_future_collision_label(
            route, current_speed, nominal_target, actors,
            safety_margin_m=args.safety_margin, include_class_ids=(1, 2),
        )
        nominal_tight = route_future_collision_label(
            route, current_speed, nominal_target, actors, safety_margin_m=0.0,
        )
        controlled = route_future_collision_label(
            route, current_speed, controlled_target, actors, safety_margin_m=args.safety_margin,
        )
        controlled_dynamic = route_future_collision_label(
            route, current_speed, controlled_target, actors,
            safety_margin_m=args.safety_margin, include_class_ids=(1, 2),
        )

        raw_positions = np.asarray(meta.get("future_positions", []), dtype=np.float32)
        raw_yaws = np.asarray(meta.get("future_yaws", []), dtype=np.float32).reshape(-1)
        indices = np.arange(5, 41, 5)
        actual_valid = (indices < len(raw_positions)) & (indices < len(raw_yaws))
        if actual_valid.all():
            actual = future_collision_label(
                raw_positions[indices, :2], raw_yaws[indices], actors,
                safety_margin_m=args.safety_margin,
            )
            actual_tight = future_collision_label(
                raw_positions[indices, :2], raw_yaws[indices], actors,
                safety_margin_m=0.0,
            )
            actual_dynamic = future_collision_label(
                raw_positions[indices, :2], raw_yaws[indices], actors,
                safety_margin_m=args.safety_margin, include_class_ids=(1, 2),
            )
        else:
            actual = None
            actual_tight = None
            actual_dynamic = None

        hazard = bool(meta.get("vehicle_hazard", False) or meta.get("walker_hazard", False))
        brake = bool(meta.get("brake", False))
        record = {
            "nominal_collision": nominal.collision,
            "nominal_dynamic_collision": nominal_dynamic.collision,
            "nominal_tight_collision": nominal_tight.collision,
            "controlled_collision": controlled.collision,
            "controlled_dynamic_collision": controlled_dynamic.collision,
            "actual_collision": actual.collision if actual is not None else False,
            "actual_dynamic_collision": actual_dynamic.collision if actual_dynamic is not None else False,
            "actual_tight_collision": actual_tight.collision if actual_tight is not None else False,
            "actual_valid": actual is not None,
            "nominal_ttc_s": nominal.ttc_s,
            "nominal_dynamic_ttc_s": nominal_dynamic.ttc_s,
            "collision_class": ACTOR_ID_TO_CLASS.get(nominal.actor_class_id, "none"),
            "controlled_collision_class": ACTOR_ID_TO_CLASS.get(controlled.actor_class_id, "none"),
            "actual_collision_class": ACTOR_ID_TO_CLASS.get(
                actual.actor_class_id if actual is not None else 0, "none",
            ),
            "hazard": hazard,
            "brake": brake,
        }
        records.append(record)
        stats = scenario_stats[item["scenario"]]
        stats["frames"] += 1
        stats["nominal_collision"] += int(nominal.collision)
        stats["hazard"] += int(hazard)
        stats["brake"] += int(brake)

    nominal = np.asarray([r["nominal_collision"] for r in records])
    nominal_dynamic = np.asarray([r["nominal_dynamic_collision"] for r in records])
    nominal_tight = np.asarray([r["nominal_tight_collision"] for r in records])
    controlled = np.asarray([r["controlled_collision"] for r in records])
    controlled_dynamic = np.asarray([r["controlled_dynamic_collision"] for r in records])
    actual_valid = np.asarray([r["actual_valid"] for r in records])
    actual = np.asarray([r["actual_collision"] for r in records])
    actual_dynamic = np.asarray([r["actual_dynamic_collision"] for r in records])
    actual_tight = np.asarray([r["actual_tight_collision"] for r in records])
    hazard = np.asarray([r["hazard"] for r in records])
    brake = np.asarray([r["brake"] for r in records])
    finite_ttc = np.asarray([r["nominal_ttc_s"] for r in records if np.isfinite(r["nominal_ttc_s"])])
    finite_dynamic_ttc = np.asarray([
        r["nominal_dynamic_ttc_s"]
        for r in records
        if np.isfinite(r["nominal_dynamic_ttc_s"])
    ])
    payload = {
        "frames": limit,
        "time_grid_s": FUTURE_TIMES_S.tolist(),
        "safety_margin_m": args.safety_margin,
        "cache": {
            "actors_by_class": dict(class_counts),
            "valid_actor_states_by_step": valid_step_counts.tolist(),
            "dropped_actors": dropped_total,
        },
        "label_rates": {
            "nominal_unbraked_collision": float(nominal.mean()),
            "nominal_unbraked_dynamic_collision": float(nominal_dynamic.mean()),
            "nominal_unbraked_collision_zero_margin": float(nominal_tight.mean()),
            "expert_controlled_route_collision": float(controlled.mean()),
            "expert_controlled_route_dynamic_collision": float(controlled_dynamic.mean()),
            "actual_ego_future_collision": float(actual[actual_valid].mean()) if actual_valid.any() else None,
            "actual_ego_future_dynamic_collision": (
                float(actual_dynamic[actual_valid].mean()) if actual_valid.any() else None
            ),
            "actual_ego_future_collision_zero_margin": (
                float(actual_tight[actual_valid].mean()) if actual_valid.any() else None
            ),
            "expert_hazard": float(hazard.mean()),
            "expert_brake": float(brake.mean()),
        },
        "nominal_dynamic_vs_expert_hazard": _binary_metrics(nominal_dynamic, hazard),
        "nominal_dynamic_vs_expert_brake": _binary_metrics(nominal_dynamic, brake),
        "nominal_all_physical_vs_expert_hazard": _binary_metrics(nominal, hazard),
        "ttc_s": {
            "count": int(len(finite_ttc)),
            "p10": float(np.quantile(finite_ttc, 0.1)) if len(finite_ttc) else None,
            "p50": float(np.quantile(finite_ttc, 0.5)) if len(finite_ttc) else None,
            "p90": float(np.quantile(finite_ttc, 0.9)) if len(finite_ttc) else None,
        },
        "dynamic_ttc_s": {
            "count": int(len(finite_dynamic_ttc)),
            "p10": float(np.quantile(finite_dynamic_ttc, 0.1)) if len(finite_dynamic_ttc) else None,
            "p50": float(np.quantile(finite_dynamic_ttc, 0.5)) if len(finite_dynamic_ttc) else None,
            "p90": float(np.quantile(finite_dynamic_ttc, 0.9)) if len(finite_dynamic_ttc) else None,
        },
        "collision_actor_class": dict(Counter(r["collision_class"] for r in records if r["nominal_collision"])),
        "controlled_collision_actor_class": dict(Counter(
            r["controlled_collision_class"] for r in records if r["controlled_collision"]
        )),
        "actual_collision_actor_class": dict(Counter(
            r["actual_collision_class"] for r in records if r["actual_collision"]
        )),
        "scenarios": {
            name: {
                **dict(counts),
                "nominal_collision_rate": counts["nominal_collision"] / counts["frames"],
                "hazard_rate": counts["hazard"] / counts["frames"],
                "brake_rate": counts["brake"] / counts["frames"],
            }
            for name, counts in sorted(scenario_stats.items())
        },
        "definition": {
            "nominal_unbraked": "GT route, speed toward max(target_speed, second_highest_speed)",
            "controlled": "GT route, speed toward expert target_speed after hazard response",
            "actual": "recorded expert future positions/yaws",
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(payload, handle, indent=2)

    rates = payload["label_rates"]
    print(f"\nB2d future collision audit ({limit} frames)")
    print(
        " collision rate nominal/controlled/actual: "
        f"{rates['nominal_unbraked_collision']:.1%} / "
        f"{rates['expert_controlled_route_collision']:.1%} / "
        f"{rates['actual_ego_future_collision']:.1%}",
    )
    print(f" expert hazard/brake: {rates['expert_hazard']:.1%} / {rates['expert_brake']:.1%}")
    print(
        " dynamic collision rate nominal/controlled/actual: "
        f"{rates['nominal_unbraked_dynamic_collision']:.1%} / "
        f"{rates['expert_controlled_route_dynamic_collision']:.1%} / "
        f"{rates['actual_ego_future_dynamic_collision']:.1%}",
    )
    print(" nominal dynamic vs hazard:", payload["nominal_dynamic_vs_expert_hazard"])
    print(" nominal dynamic TTC:", payload["dynamic_ttc_s"])
    print(" collision classes:", payload["collision_actor_class"])
    print(f" wrote {output}")


if __name__ == "__main__":
    main()
