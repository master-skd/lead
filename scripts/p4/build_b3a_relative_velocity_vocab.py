"""Build a current-speed-relative velocity-profile vocabulary from expert data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def realize_relative_profiles(
    relative_centers: np.ndarray, current_speeds: np.ndarray
) -> np.ndarray:
    """Turn delta-speed centers into non-negative absolute profiles per frame."""

    centers = np.asarray(relative_centers, dtype=np.float32)
    current = np.maximum(np.asarray(current_speeds, dtype=np.float32).reshape(-1), 0.0)
    if centers.ndim != 2:
        raise ValueError("relative centers must have shape [K, T]")
    return np.maximum(current[:, None, None] + centers[None], 0.0).astype(np.float32)


def nearest_relative_profiles(
    expert_profiles: np.ndarray,
    current_speeds: np.ndarray,
    centers: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    assignments = np.empty(len(expert_profiles), dtype=np.int32)
    error = np.empty(len(expert_profiles), dtype=np.float32)
    for start in range(0, len(expert_profiles), chunk_size):
        end = min(start + chunk_size, len(expert_profiles))
        realized = realize_relative_profiles(centers, current_speeds[start:end])
        distance = np.abs(expert_profiles[start:end, None] - realized).mean(axis=2)
        local = distance.argmin(axis=1)
        assignments[start:end] = local
        error[start:end] = distance[np.arange(end - start), local]
    return assignments, error


def sort_relative_centers(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float32)
    order = np.lexsort((centers[:, -1], centers.sum(axis=1)))
    return centers[order]


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.quantile(values, q))
        for name, q in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99))
    }


def summarize(
    arrays: dict[str, np.ndarray], centers: np.ndarray, interval_s: float
) -> tuple[dict[str, Any], np.ndarray]:
    assignments, error = nearest_relative_profiles(
        arrays["profiles"], arrays["current_speed"], centers
    )
    realized = realize_relative_profiles(centers, arrays["current_speed"])
    rows = np.arange(len(assignments))
    chosen = realized[rows, assignments]
    expert_progress = arrays["profiles"].sum(axis=1) * interval_s
    chosen_progress = chosen.sum(axis=1) * interval_s
    terminal = arrays["profiles"][:, -1]
    current = arrays["current_speed"]
    stopped = terminal <= 0.3
    behavior = {
        "stopped": stopped,
        "braking": (~stopped) & (terminal - current <= -1.0),
        "accelerating": terminal - current >= 1.0,
        "steady": (~stopped) & (terminal - current > -1.0) & (terminal - current < 1.0),
    }
    return {
        "n": len(error),
        "nearest_profile_mae_mps": {"mean": float(error.mean()), **_quantiles(error)},
        "coverage": {
            f"mae_le_{threshold:g}_mps": float((error <= threshold).mean())
            for threshold in (0.25, 0.5, 1.0, 2.0)
        },
        "two_second_progress_abs_error_m": {
            "mean": float(np.abs(expert_progress - chosen_progress).mean()),
            **_quantiles(np.abs(expert_progress - chosen_progress)),
        },
        "per_step_mae_mps": np.abs(arrays["profiles"] - chosen).mean(axis=0).tolist(),
        "behavior": {
            name: {
                "n": int(mask.sum()),
                "fraction": float(mask.mean()),
                "nearest_profile_mae_mean_mps": float(error[mask].mean())
                if mask.any()
                else None,
                "nearest_profile_mae_p90_mps": float(np.quantile(error[mask], 0.9))
                if mask.any()
                else None,
            }
            for name, mask in behavior.items()
        },
    }, assignments


def center_summary(
    centers: np.ndarray, assignments: np.ndarray, interval_s: float
) -> list[dict[str, Any]]:
    support = np.bincount(assignments, minlength=len(centers))
    first_acceleration = 2.0 * centers[:, 0] / interval_s
    transition_acceleration = np.diff(centers, axis=1) / interval_s
    return [
        {
            "index": index,
            "support": int(support[index]),
            "support_fraction": float(support[index] / max(len(assignments), 1)),
            "mean_delta_speed_mps": float(center.mean()),
            "terminal_delta_speed_mps": float(center[-1]),
            "delta_progress_m": float(center.sum() * interval_s),
            "first_interval_acceleration_mps2": float(first_acceleration[index]),
            "transition_acceleration_min_mps2": float(
                transition_acceleration[index].min()
            ),
            "transition_acceleration_max_mps2": float(
                transition_acceleration[index].max()
            ),
            "delta_profile_mps": center.tolist(),
        }
        for index, center in enumerate(centers)
    ]


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        return {
            key: cache[key]
            for key in ("keys", "profiles", "current_speed", "target_speed", "brake")
        }


def _parse_clusters(value: str) -> tuple[int, ...]:
    result = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not result or result[0] < 2:
        raise ValueError("--clusters must contain comma-separated integers >= 2")
    return result


def _plot(vocabularies: dict[int, np.ndarray], interval_s: float, path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, len(vocabularies), figsize=(6 * len(vocabularies), 5), squeeze=False
    )
    for axis, (clusters, centers) in zip(
        axes[0], sorted(vocabularies.items()), strict=True
    ):
        times = np.arange(1, centers.shape[1] + 1) * interval_s
        colors = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(centers)))
        for center, color in zip(centers, colors, strict=True):
            axis.plot(times, center, color=color, alpha=0.8, linewidth=1.0)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"K={clusters}")
        axis.set_xlabel("future time (s)")
        axis.set_ylabel("speed residual from current speed (m/s)")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clusters", default="16,32,64")
    parser.add_argument("--interval-s", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _load_cache(source_dir / "train_velocity_profiles.npz")
    heldout = _load_cache(source_dir / "heldout_velocity_profiles.npz")
    clusters = _parse_clusters(args.clusters)
    train_relative = train["profiles"] - train["current_speed"][:, None]

    from sklearn.cluster import KMeans

    results: dict[str, Any] = {}
    vocabularies: dict[int, np.ndarray] = {}
    for cluster_count in clusters:
        print(
            f"fitting relative K-means K={cluster_count} on "
            f"{len(train_relative)} expert profiles"
        )
        estimator = KMeans(
            n_clusters=cluster_count,
            random_state=args.seed,
            n_init=args.n_init,
            max_iter=args.max_iter,
            algorithm="lloyd",
        ).fit(train_relative)
        centers = sort_relative_centers(estimator.cluster_centers_)
        train_summary, train_assignment = summarize(train, centers, args.interval_s)
        heldout_summary, heldout_assignment = summarize(
            heldout, centers, args.interval_s
        )
        vocabularies[cluster_count] = centers
        np.save(output_dir / f"relative_velocity_vocab_k{cluster_count}.npy", centers)
        np.savez_compressed(
            output_dir / f"relative_velocity_vocab_k{cluster_count}_assignments.npz",
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
            f"K={cluster_count}: held-out MAE="
            f"{heldout_summary['nearest_profile_mae_mps']['mean']:.3f} m/s "
            f"p90={heldout_summary['nearest_profile_mae_mps']['p90']:.3f} "
            f"coverage@0.5={heldout_summary['coverage']['mae_le_0.5_mps']:.1%}"
        )

    payload = {
        "source_dir": str(source_dir.resolve()),
        "train_frames": len(train["profiles"]),
        "heldout_frames": len(heldout["profiles"]),
        "representation": {
            "definition": "delta_v[t] = expert_interval_average_speed[t] - current_speed",
            "realization": "candidate_speed[t] = max(0, current_speed + delta_vocabulary[k,t])",
            "steps": train["profiles"].shape[1],
            "interval_s": args.interval_s,
            "units": "m/s",
        },
        "kmeans": {
            "fit_split": "same route-disjoint train split as absolute vocabulary",
            "seed": args.seed,
            "n_init": args.n_init,
            "max_iter": args.max_iter,
            "cluster_counts": list(clusters),
        },
        "results": results,
    }
    audit_path = output_dir / "relative_velocity_vocab_audit.json"
    with audit_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    if not args.skip_plot:
        _plot(vocabularies, args.interval_s, output_dir / "relative_velocity_vocab.png")
    print(f"wrote {audit_path}")


if __name__ == "__main__":
    main()
