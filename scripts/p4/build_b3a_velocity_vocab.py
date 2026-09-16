"""Build and audit an expert velocity-profile vocabulary for P5 B3a.

The source CARLA expert metadata stores ego speed at 20 Hz in ``future_speeds``.
This script converts it to interval-average speeds on the planner's 4 Hz / 2 s
grid, fits K-means on route-disjoint training frames only, and reports held-out
quantization coverage.  It intentionally does not load or change the LEAD model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import pickle
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


def interval_average_profile(
    future_speeds: np.ndarray,
    *,
    steps: int = 8,
    interval_s: float = 0.25,
    raw_dt_s: float = 0.05,
) -> np.ndarray:
    """Convert point speeds to interval averages using trapezoidal integration."""

    stride_float = interval_s / raw_dt_s
    stride = int(round(stride_float))
    if stride < 1 or not np.isclose(stride, stride_float, atol=1e-6):
        raise ValueError("interval_s must be an integer multiple of raw_dt_s")
    values = np.asarray(future_speeds, dtype=np.float32).reshape(-1)
    required = steps * stride + 1
    if len(values) < required:
        raise ValueError(f"future_speeds has {len(values)} samples; need {required}")
    values = values[:required]
    if not np.isfinite(values).all():
        raise ValueError("future_speeds contains a non-finite value")

    # Integral / interval duration. Written explicitly instead of np.trapezoid so
    # this remains compatible with the NumPy version in the LEAD environment.
    profile = np.empty(steps, dtype=np.float32)
    for step in range(steps):
        segment = values[step * stride : (step + 1) * stride + 1]
        profile[step] = (
            0.5 * float(segment[0])
            + float(segment[1:-1].sum())
            + 0.5 * float(segment[-1])
        ) / stride
    return np.maximum(profile, 0.0)


def _meta_path(entry: dict[str, Any], repo_root: Path) -> Path:
    source = Path(entry["src"])
    if not source.is_absolute():
        source = repo_root / source
    return source.parent.parent / "metas" / f"{entry['frame']}.pkl"


def _extract_one(
    item: tuple[dict[str, Any], str, int, float, float, float, Path],
) -> tuple[str, tuple[Any, ...] | None]:
    entry, key, steps, interval_s, raw_dt_s, max_speed_mps, repo_root = item
    path = _meta_path(entry, repo_root)
    try:
        with lzma.open(path, "rb") as handle:
            meta = pickle.load(handle)
    except (EOFError, OSError, lzma.LZMAError, pickle.UnpicklingError):
        return "unreadable_meta", None
    if "future_speeds" not in meta:
        return "missing_future_speeds", None
    try:
        profile = interval_average_profile(
            meta["future_speeds"],
            steps=steps,
            interval_s=interval_s,
            raw_dt_s=raw_dt_s,
        )
    except ValueError as error:
        reason = "incomplete_future" if "need" in str(error) else "invalid_future"
        return reason, None
    current_speed = float(meta.get("speed", np.asarray(meta["future_speeds"])[0]))
    target_speed = float(meta.get("target_speed", current_speed))
    if not np.isfinite(current_speed) or not np.isfinite(target_speed):
        return "invalid_scalar_speed", None
    current_speed = max(0.0, current_speed)
    target_speed = max(0.0, target_speed)
    if max(float(profile.max()), current_speed, target_speed) > max_speed_mps:
        return "speed_out_of_range", None
    return "ok", (
        key,
        profile,
        current_speed,
        target_speed,
        bool(meta.get("brake", False)),
    )


def read_manifest(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    with open(path) as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    if limit is not None:
        entries = entries[: max(0, limit)]
    keys = [str(entry["key"]) for entry in entries]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate keys in {path}")
    return entries


def extract_profiles(
    entries: list[dict[str, Any]],
    *,
    repo_root: Path,
    steps: int,
    interval_s: float,
    raw_dt_s: float,
    max_speed_mps: float,
    workers: int,
    description: str,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    work = [
        (
            entry,
            str(entry["key"]),
            steps,
            interval_s,
            raw_dt_s,
            max_speed_mps,
            repo_root,
        )
        for entry in entries
    ]
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(
                tqdm(pool.map(_extract_one, work), total=len(work), desc=description)
            )
    else:
        results = [_extract_one(item) for item in tqdm(work, desc=description)]

    reasons = Counter(reason for reason, _ in results)
    rows = [row for reason, row in results if reason == "ok" and row is not None]
    if not rows:
        raise ValueError(f"no complete velocity profiles extracted from {description}")
    arrays = {
        "keys": np.asarray([row[0] for row in rows]),
        "profiles": np.stack([row[1] for row in rows]).astype(np.float32),
        "current_speed": np.asarray([row[2] for row in rows], dtype=np.float32),
        "target_speed": np.asarray([row[3] for row in rows], dtype=np.float32),
        "brake": np.asarray([row[4] for row in rows], dtype=bool),
    }
    return arrays, dict(sorted(reasons.items()))


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _nearest(
    profiles: np.ndarray,
    centers: np.ndarray,
    chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    assignments = np.empty(len(profiles), dtype=np.int32)
    mean_absolute_error = np.empty(len(profiles), dtype=np.float32)
    for start in range(0, len(profiles), chunk_size):
        end = min(start + chunk_size, len(profiles))
        distance = np.abs(profiles[start:end, None] - centers[None]).mean(axis=-1)
        local = distance.argmin(axis=1)
        assignments[start:end] = local
        mean_absolute_error[start:end] = distance[np.arange(end - start), local]
    return assignments, mean_absolute_error


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.quantile(values, q))
        for name, q in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99))
    }


def behavior_masks(
    profiles: np.ndarray,
    current_speed: np.ndarray,
) -> dict[str, np.ndarray]:
    terminal = profiles[:, -1]
    delta = terminal - current_speed
    stopped = terminal <= 0.3
    return {
        "stopped": stopped,
        "braking": (~stopped) & (delta <= -1.0),
        "accelerating": delta >= 1.0,
        "steady": (~stopped) & (delta > -1.0) & (delta < 1.0),
    }


def summarize_quantization(
    arrays: dict[str, np.ndarray],
    centers: np.ndarray,
    interval_s: float,
) -> tuple[dict[str, Any], np.ndarray]:
    assignments, error = _nearest(arrays["profiles"], centers)
    reconstructed = centers[assignments]
    progress = arrays["profiles"].sum(axis=1) * interval_s
    reconstructed_progress = reconstructed.sum(axis=1) * interval_s
    summary: dict[str, Any] = {
        "n": len(error),
        "nearest_profile_mae_mps": {
            "mean": float(error.mean()),
            **_quantiles(error),
        },
        "coverage": {
            f"mae_le_{threshold:g}_mps": float((error <= threshold).mean())
            for threshold in (0.25, 0.5, 1.0, 2.0)
        },
        "two_second_progress_abs_error_m": {
            "mean": float(np.abs(progress - reconstructed_progress).mean()),
            **_quantiles(np.abs(progress - reconstructed_progress)),
        },
        "per_step_mae_mps": np.abs(arrays["profiles"] - reconstructed)
        .mean(axis=0)
        .tolist(),
        "behavior": {},
    }
    for name, mask in behavior_masks(
        arrays["profiles"], arrays["current_speed"]
    ).items():
        summary["behavior"][name] = {
            "n": int(mask.sum()),
            "fraction": float(mask.mean()),
            "nearest_profile_mae_mean_mps": float(error[mask].mean())
            if mask.any()
            else None,
            "nearest_profile_mae_p90_mps": float(np.quantile(error[mask], 0.9))
            if mask.any()
            else None,
        }
    return summary, assignments


def sort_centers(centers: np.ndarray) -> np.ndarray:
    """Give arbitrary K-means clusters a stable slow-to-fast vocabulary order."""

    progress = centers.sum(axis=1)
    order = np.lexsort((centers[:, -1], progress))
    return np.maximum(centers[order], 0.0).astype(np.float32)


def center_summary(
    centers: np.ndarray,
    assignments: np.ndarray,
    interval_s: float,
) -> list[dict[str, Any]]:
    support = np.bincount(assignments, minlength=len(centers))
    rows = []
    for index, center in enumerate(centers):
        rows.append(
            {
                "index": index,
                "support": int(support[index]),
                "support_fraction": float(support[index] / max(len(assignments), 1)),
                "mean_speed_mps": float(center.mean()),
                "terminal_speed_mps": float(center[-1]),
                "progress_m": float(center.sum() * interval_s),
                "profile_mps": center.tolist(),
            }
        )
    return rows


def plot_vocabularies(
    vocabularies: dict[int, np.ndarray], interval_s: float, path: Path
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, len(vocabularies), figsize=(6 * len(vocabularies), 5), squeeze=False
    )
    for axis, (clusters, centers) in zip(
        axes[0], sorted(vocabularies.items()), strict=True
    ):
        times = np.arange(1, centers.shape[1] + 1) * interval_s
        colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(centers)))
        for center, color in zip(centers, colors, strict=True):
            axis.plot(times, center, color=color, alpha=0.8, linewidth=1.0)
        axis.set_title(f"K={clusters}")
        axis.set_xlabel("future time (s)")
        axis.set_ylabel("interval-average speed (m/s)")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_clusters(value: str) -> tuple[int, ...]:
    clusters = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not clusters or clusters[0] < 2:
        raise ValueError("--clusters must contain integers >= 2")
    return clusters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-manifest",
        default="outputs/local_training/p5_stepB2_corridor/b2d_route_split/train.jsonl",
    )
    parser.add_argument(
        "--heldout-manifest",
        default="outputs/local_training/p5_stepB2_corridor/b2d_route_split/heldout.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/local_training/p5_stepB3a_velocity_vocab",
    )
    parser.add_argument("--clusters", default="16,32,64")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--interval-s", type=float, default=0.25)
    parser.add_argument("--raw-dt-s", type=float, default=0.05)
    parser.add_argument("--max-speed-mps", type=float, default=40.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-heldout", type=int, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    clusters = parse_clusters(args.clusters)

    train_manifest = Path(args.train_manifest)
    heldout_manifest = Path(args.heldout_manifest)
    if not train_manifest.is_absolute():
        train_manifest = repo_root / train_manifest
    if not heldout_manifest.is_absolute():
        heldout_manifest = repo_root / heldout_manifest
    train_entries = read_manifest(train_manifest, args.limit_train)
    heldout_entries = read_manifest(heldout_manifest, args.limit_heldout)
    train_routes = {(entry["scenario"], entry["route"]) for entry in train_entries}
    heldout_routes = {(entry["scenario"], entry["route"]) for entry in heldout_entries}
    overlap = train_routes & heldout_routes
    if overlap:
        raise ValueError(f"train/held-out route leakage: first={sorted(overlap)[0]}")

    representation_metadata = {
        "steps": args.steps,
        "interval_s": args.interval_s,
        "raw_dt_s": args.raw_dt_s,
        "max_speed_mps": args.max_speed_mps,
    }

    def load_or_extract(name: str, entries: list[dict[str, Any]]):
        path = output_dir / f"{name}_velocity_profiles.npz"
        entry_keys = [str(entry["key"]) for entry in entries]
        expected_metadata = {
            **representation_metadata,
            "manifest_frames": len(entries),
            "manifest_keys_sha256": hashlib.sha256(
                "\n".join(entry_keys).encode(),
            ).hexdigest(),
        }
        if path.exists() and not args.overwrite_cache:
            with np.load(path, allow_pickle=False) as cache:
                actual = json.loads(str(cache["metadata_json"]))
                if actual != expected_metadata:
                    raise ValueError(
                        f"cached representation mismatch in {path}; use --overwrite-cache"
                    )
                arrays = {
                    key: cache[key]
                    for key in (
                        "keys",
                        "profiles",
                        "current_speed",
                        "target_speed",
                        "brake",
                    )
                }
                extraction = json.loads(str(cache["extraction_json"]))
            print(f"loaded {path}: {len(arrays['keys'])} complete profiles")
            return arrays, extraction
        arrays, extraction = extract_profiles(
            entries,
            repo_root=repo_root,
            steps=args.steps,
            interval_s=args.interval_s,
            raw_dt_s=args.raw_dt_s,
            max_speed_mps=args.max_speed_mps,
            workers=args.workers,
            description=f"extract {name} velocity",
        )
        save_npz_atomic(
            path,
            **arrays,
            metadata_json=np.asarray(json.dumps(expected_metadata, sort_keys=True)),
            extraction_json=np.asarray(json.dumps(extraction, sort_keys=True)),
        )
        print(f"wrote {path}: {len(arrays['keys'])} complete profiles")
        return arrays, extraction

    train, train_extraction = load_or_extract("train", train_entries)
    heldout, heldout_extraction = load_or_extract("heldout", heldout_entries)
    if len(train["profiles"]) < max(clusters):
        raise ValueError(
            "fewer complete training profiles than the requested largest K"
        )

    from sklearn.cluster import KMeans

    vocabularies: dict[int, np.ndarray] = {}
    results: dict[str, Any] = {}
    for cluster_count in clusters:
        print(
            f"fitting K-means K={cluster_count} on {len(train['profiles'])} expert profiles"
        )
        estimator = KMeans(
            n_clusters=cluster_count,
            random_state=args.seed,
            n_init=args.n_init,
            max_iter=args.max_iter,
            algorithm="lloyd",
        ).fit(train["profiles"])
        centers = sort_centers(estimator.cluster_centers_)
        vocabularies[cluster_count] = centers
        train_summary, train_assignment = summarize_quantization(
            train, centers, args.interval_s
        )
        heldout_summary, heldout_assignment = summarize_quantization(
            heldout, centers, args.interval_s
        )
        np.save(output_dir / f"velocity_vocab_k{cluster_count}.npy", centers)
        save_npz_atomic(
            output_dir / f"velocity_vocab_k{cluster_count}_assignments.npz",
            train_keys=train["keys"],
            train_assignment=train_assignment,
            heldout_keys=heldout["keys"],
            heldout_assignment=heldout_assignment,
        )
        results[str(cluster_count)] = {
            "inertia": float(estimator.inertia_),
            "iterations": int(estimator.n_iter_),
            "train": train_summary,
            "heldout": heldout_summary,
            "centers": center_summary(centers, train_assignment, args.interval_s),
        }
        print(
            f"K={cluster_count}: held-out MAE={heldout_summary['nearest_profile_mae_mps']['mean']:.3f} m/s "
            f"p90={heldout_summary['nearest_profile_mae_mps']['p90']:.3f} "
            f"coverage@0.5={heldout_summary['coverage']['mae_le_0.5_mps']:.1%}"
        )

    payload = {
        "train_manifest": str(train_manifest.resolve()),
        "heldout_manifest": str(heldout_manifest.resolve()),
        "route_overlap": 0,
        "representation": {
            **representation_metadata,
            "times_s": (np.arange(1, args.steps + 1) * args.interval_s).tolist(),
            "definition": "trapezoidal interval-average speed from 20 Hz expert future_speeds",
            "units": "m/s",
            "training_representation": "absolute ego speed",
        },
        "extraction": {
            "train_manifest_frames": len(train_entries),
            "heldout_manifest_frames": len(heldout_entries),
            "train": train_extraction,
            "heldout": heldout_extraction,
        },
        "kmeans": {
            "fit_split": "train routes only",
            "seed": args.seed,
            "n_init": args.n_init,
            "max_iter": args.max_iter,
            "cluster_counts": list(clusters),
        },
        "results": results,
    }
    audit_path = output_dir / "velocity_vocab_audit.json"
    with audit_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    if not args.skip_plot:
        plot_vocabularies(
            vocabularies, args.interval_s, output_dir / "velocity_vocab.png"
        )
    print(f"wrote {audit_path}")


if __name__ == "__main__":
    main()
