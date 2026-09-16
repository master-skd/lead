"""Evaluate K-means velocity vocabularies on fixed LEAD spatial routes.

This is a no-training, offline upper-bound audit.  It reuses the frame records
written by ``eval_b3a_velocity_profile_oracle.py`` so the route, raw velocity
profile, held-out frames, and GT future actors exactly match B3a-0b.  Each
vocabulary profile is integrated into arc length, interpolated on the fixed
confidence-selected route, and assigned counterfactual collision labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


def parse_clusters(value: str) -> tuple[int, ...]:
    clusters = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not clusters or any(item < 2 for item in clusters):
        raise ValueError("--clusters must contain comma-separated integers >= 2")
    return clusters


def profile_distances(
    interval_average_speeds: np.ndarray, interval_s: float
) -> np.ndarray:
    """Integrate interval-average speed profiles into end-of-interval distances."""

    profiles = np.asarray(interval_average_speeds, dtype=np.float32)
    if profiles.ndim < 1:
        raise ValueError("speed profiles must have at least one dimension")
    if interval_s <= 0:
        raise ValueError("interval_s must be positive")
    if not np.isfinite(profiles).all() or (profiles < 0).any():
        raise ValueError("speed profiles must be finite and non-negative")
    return np.cumsum(profiles, axis=-1, dtype=np.float32) * float(interval_s)


def interval_speeds_from_distances(
    distances: np.ndarray, interval_s: float
) -> np.ndarray:
    """Recover interval-average speeds from cumulative distances."""

    values = np.asarray(distances, dtype=np.float32)
    zeros = np.zeros((*values.shape[:-1], 1), dtype=np.float32)
    return np.diff(np.concatenate((zeros, values), axis=-1), axis=-1) / interval_s


def start_reachability_mask(
    profiles: np.ndarray,
    current_speeds: np.ndarray,
    interval_s: float,
    *,
    max_accel_mps2: float = 1.89,
    max_decel_mps2: float = 4.95,
) -> np.ndarray:
    """Approximate whether the first interval is reachable from current speed.

    For a constant acceleration over the first interval, its average speed is
    ``v0 + 0.5 * a * dt``.  Later transitions are reported separately because a
    K-means center is an interval-average representation rather than a piecewise
    constant-acceleration control signal.
    """

    profiles = np.asarray(profiles, dtype=np.float32)
    current = np.maximum(np.asarray(current_speeds, dtype=np.float32).reshape(-1), 0.0)
    if profiles.ndim == 2:
        first_speed = profiles[None, :, 0]
    elif profiles.ndim == 3 and profiles.shape[0] == len(current):
        first_speed = profiles[:, :, 0]
    else:
        raise ValueError("profiles must have shape [K, T] or [N, K, T]")
    if profiles.shape[-1] == 0:
        raise ValueError("profiles must contain at least one future step")
    first_acceleration = 2.0 * (first_speed - current[:, None]) / interval_s
    return (first_acceleration >= -max_decel_mps2 - 1e-6) & (
        first_acceleration <= max_accel_mps2 + 1e-6
    )


def transition_reachability_mask(
    profiles: np.ndarray,
    interval_s: float,
    *,
    max_accel_mps2: float = 1.89,
    max_decel_mps2: float = 4.95,
) -> np.ndarray:
    """Flag centers whose adjacent interval-average speeds respect accel bounds."""

    profiles = np.asarray(profiles, dtype=np.float32)
    acceleration = np.diff(profiles, axis=-1) / interval_s
    return (
        (acceleration >= -max_decel_mps2 - 1e-6)
        & (acceleration <= max_accel_mps2 + 1e-6)
    ).all(axis=-1)


def select_progress_oracle(
    raw_collision: np.ndarray,
    candidate_collision: np.ndarray,
    candidate_valid: np.ndarray,
    candidate_progress: np.ndarray,
) -> np.ndarray:
    """Keep a safe raw trajectory; otherwise select max-progress safe vocabulary.

    Returned index zero denotes the raw trajectory. Vocabulary index ``j`` is
    returned as ``j + 1``.
    """

    raw_collision = np.asarray(raw_collision, dtype=bool).reshape(-1)
    collision = np.asarray(candidate_collision, dtype=bool)
    valid = np.asarray(candidate_valid, dtype=bool)
    progress = np.asarray(candidate_progress, dtype=np.float32)
    if collision.shape != valid.shape or collision.shape[0] != len(raw_collision):
        raise ValueError("collision/valid/raw shapes do not agree")
    if progress.ndim == 1:
        if collision.shape[1] != len(progress):
            raise ValueError("candidate progress does not match candidate count")
        progress = np.broadcast_to(progress[None], collision.shape)
    elif progress.shape != collision.shape:
        raise ValueError("candidate progress must have shape [K] or [N, K]")

    safe = valid & ~collision
    score = np.where(safe, progress, -np.inf)
    best = score.argmax(axis=1)
    has_safe = safe.any(axis=1)
    selected = np.zeros(len(raw_collision), dtype=np.int16)
    intervene = raw_collision & has_safe
    selected[intervene] = best[intervene].astype(np.int16) + 1
    return selected


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"p10": None, "p50": None, "p90": None}
    return {
        name: float(np.quantile(values, q))
        for name, q in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9))
    }


def _masked_nearest(error: np.ndarray, valid: np.ndarray) -> np.ndarray:
    masked = np.where(valid, error, np.inf)
    nearest = masked.min(axis=1)
    nearest[~valid.any(axis=1)] = np.nan
    return nearest


def summarize_policy(
    arrays: dict[str, np.ndarray],
    margin_index: int,
    scope: np.ndarray,
    policy_valid: np.ndarray,
) -> dict[str, Any]:
    """Summarize a vocabulary policy for one margin and evaluation scope."""

    rows = np.flatnonzero(scope)
    raw_collision = arrays["raw_collision"][rows, margin_index]
    collision = arrays["vocab_collision"][rows, margin_index]
    valid = policy_valid[rows]
    vocab_distance = arrays["vocab_distance"]
    if vocab_distance.ndim == 2:
        progress = np.broadcast_to(vocab_distance[None, :, -1], collision.shape)
    else:
        progress = vocab_distance[rows, :, -1]
    raw_progress = arrays["raw_distance"][rows, -1]
    selected = select_progress_oracle(raw_collision, collision, valid, progress)
    safe = valid & ~collision
    rescued = raw_collision & safe.any(axis=1)
    unresolved = raw_collision & ~safe.any(axis=1)
    intervention = selected > 0

    selected_progress = raw_progress.copy()
    selected_terminal = arrays["raw_velocity"][rows, -1].copy()
    selected_expert_mae = arrays["raw_expert_mae"][rows].copy()
    chosen = selected[intervention] - 1
    selected_progress[intervention] = progress[intervention, chosen]
    vocab_velocity = arrays["vocab_velocity"]
    if vocab_velocity.ndim == 2:
        selected_terminal[intervention] = vocab_velocity[chosen, -1]
    else:
        selected_terminal[intervention] = vocab_velocity[rows[intervention], chosen, -1]
    selected_expert_mae[intervention] = arrays["vocab_expert_mae"][
        rows[intervention], chosen
    ]

    retained = selected_progress[rescued] / np.maximum(raw_progress[rescued], 1e-6)
    nearest = _masked_nearest(arrays["vocab_expert_mae"][rows], valid)
    valid_collision_count = int((collision & valid).sum())
    valid_count = int(valid.sum())
    vocab_family = arrays["vocab_family"]
    if vocab_family.ndim == 1:
        selected_families = vocab_family[chosen]
    else:
        selected_families = vocab_family[rows[intervention], chosen]
    return {
        "n": int(len(rows)),
        "raw_collision_rate": float(raw_collision.mean()),
        "raw_collision_count": int(raw_collision.sum()),
        "rescue_count": int(rescued.sum()),
        "rescue_rate_given_raw_collision": _safe_div(
            int(rescued.sum()), int(raw_collision.sum())
        ),
        "all_candidates_unsafe_count": int(unresolved.sum()),
        "oracle_collision_rate": float(unresolved.mean()),
        "oracle_intervention_rate": float(intervention.mean()),
        "moving_rescue_rate_given_raw_collision": _safe_div(
            int((rescued & (selected_terminal > 0.3)).sum()), int(raw_collision.sum())
        ),
        "stopping_rescue_rate_given_raw_collision": _safe_div(
            int((rescued & (selected_terminal <= 0.3)).sum()), int(raw_collision.sum())
        ),
        "candidate_count": {
            "mean": float(valid.sum(axis=1).mean()),
            "min": int(valid.sum(axis=1).min()),
            "max": int(valid.sum(axis=1).max()),
            "zero_candidate_rate": float((~valid.any(axis=1)).mean()),
        },
        "candidate_collision_rate": _safe_div(valid_collision_count, valid_count),
        "route_overflow_rate": _safe_div(
            int((arrays["vocab_overflow"][rows] & valid).sum()), valid_count
        ),
        "nearest_expert_profile_mae_mps": {
            "mean": float(np.nanmean(nearest)) if np.isfinite(nearest).any() else None,
            **_quantiles(nearest),
        },
        "selected_expert_profile_mae_mps": {
            "mean": float(selected_expert_mae.mean()),
            **_quantiles(selected_expert_mae),
        },
        "progress_m": {
            "raw_mean": float(raw_progress.mean()),
            "selected_mean": float(selected_progress.mean()),
            "reduction_on_rescued": _quantiles(
                raw_progress[rescued] - selected_progress[rescued]
            ),
            "retained_fraction_on_rescued": _quantiles(retained),
        },
        "selected_vocab_family_on_rescued": dict(Counter(selected_families.tolist())),
    }


def summarize_manual_union(
    base: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    margin_index: int,
    scope: np.ndarray,
    policy_valid: np.ndarray,
) -> dict[str, Any]:
    """Measure incremental rescue from adding valid vocabulary to manual profiles."""

    rows = np.flatnonzero(scope)
    raw_collision = arrays["raw_collision"][rows, margin_index]
    manual_valid = base["candidate_valid"][rows]
    manual_collision = base["candidate_collision"][rows, margin_index]
    manual_safe = (manual_valid & ~manual_collision).any(axis=1)
    vocab_valid = policy_valid[rows]
    vocab_collision = arrays["vocab_collision"][rows, margin_index]
    vocab_safe = (vocab_valid & ~vocab_collision).any(axis=1)
    manual_rescued = raw_collision & manual_safe
    vocab_rescued = raw_collision & vocab_safe
    union_rescued = raw_collision & (manual_safe | vocab_safe)
    incremental = raw_collision & ~manual_safe & vocab_safe
    unresolved = raw_collision & ~manual_safe & ~vocab_safe
    return {
        "n": int(len(rows)),
        "raw_collision_count": int(raw_collision.sum()),
        "manual_rescue_count": int(manual_rescued.sum()),
        "manual_rescue_rate_given_raw_collision": _safe_div(
            int(manual_rescued.sum()), int(raw_collision.sum())
        ),
        "vocabulary_rescue_count": int(vocab_rescued.sum()),
        "vocabulary_rescue_rate_given_raw_collision": _safe_div(
            int(vocab_rescued.sum()), int(raw_collision.sum())
        ),
        "union_rescue_count": int(union_rescued.sum()),
        "union_rescue_rate_given_raw_collision": _safe_div(
            int(union_rescued.sum()), int(raw_collision.sum())
        ),
        "incremental_vocabulary_rescue_count": int(incremental.sum()),
        "incremental_vocabulary_rescue_rate_given_raw_collision": _safe_div(
            int(incremental.sum()), int(raw_collision.sum())
        ),
        "all_union_candidates_unsafe_count": int(unresolved.sum()),
        "union_oracle_collision_rate": float(unresolved.mean()),
    }


def _behavior_scopes(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    expert_terminal = arrays["expert_velocity"][:, -1]
    current = arrays["current_speed"]
    stopped = expert_terminal <= 0.3
    return {
        "all": np.ones(len(current), dtype=bool),
        "multi": arrays["multi"],
        "expert_brake": arrays["expert_brake"],
        "expert_go": ~arrays["expert_brake"],
        "expert_stopped": stopped,
        "expert_braking": (~stopped) & (expert_terminal - current <= -1.0),
        "expert_accelerating": expert_terminal - current >= 1.0,
        "expert_steady": (~stopped)
        & (expert_terminal - current > -1.0)
        & (expert_terminal - current < 1.0),
    }


def _load_actor_cache(
    cache_dir: Path,
) -> tuple[dict[Path, Any], dict[str, tuple[Path, int]]]:
    shards: dict[Path, Any] = {}
    locations: dict[str, tuple[Path, int]] = {}
    for path in sorted(cache_dir.glob("future_actors_*.npz")):
        shard = np.load(path, allow_pickle=False)
        shards[path] = shard
        for index, key in enumerate(shard["keys"]):
            locations[str(key)] = (path, index)
    return shards, locations


def _load_expert_profiles(path: Path) -> dict[str, tuple[np.ndarray, float]]:
    with np.load(path, allow_pickle=False) as cache:
        return {
            str(key): (profile.astype(np.float32), float(current_speed))
            for key, profile, current_speed in zip(
                cache["keys"], cache["profiles"], cache["current_speed"], strict=True
            )
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-frames", required=True)
    parser.add_argument("--base-report")
    parser.add_argument("--vocab-dir", required=True)
    parser.add_argument("--expert-profiles", required=True)
    parser.add_argument("--future-cache-dir", required=True)
    parser.add_argument("--clusters", default="16,32,64")
    parser.add_argument(
        "--representation",
        choices=("absolute", "current-relative"),
        default="absolute",
    )
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument(
        "--collision-scope", choices=("dynamic", "all"), default="dynamic"
    )
    parser.add_argument(
        "--route-end-policy", choices=("extrapolate", "clip"), default="extrapolate"
    )
    parser.add_argument("--max-accel", type=float, default=1.89)
    parser.add_argument("--max-decel", type=float, default=4.95)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from lead.data_loader.future_actor_cache import unpack_future_actor_frame
    from lead.tfv6.future_collision import (
        future_collision_label,
        interpolate_route_by_distance,
    )
    from scripts.p4.eval_b3a_velocity_profile_oracle import summarize_margin

    if args.representation == "current-relative":
        from scripts.p4.build_b3a_relative_velocity_vocab import (
            realize_relative_profiles,
        )

    clusters = parse_clusters(args.clusters)
    vocab_dir = Path(args.vocab_dir)
    base_path = Path(args.base_frames)
    with np.load(base_path, allow_pickle=False) as source:
        total = len(source["key"])
        n = min(total, args.limit) if args.limit is not None else total
        base = {
            key: source[key][:n]
            for key in source.files
            if key not in {"margins_m", "future_times_s"}
        }
        margins = source["margins_m"].astype(np.float32)
        times = source["future_times_s"].astype(np.float32)
    if n == 0:
        raise ValueError("base frame file contains no evaluation frames")
    if len(times) < 1 or not np.allclose(np.diff(times), times[0], atol=1e-6):
        raise ValueError(
            "future time grid must be uniformly spaced and start at one interval"
        )
    interval_s = float(times[0])

    if args.base_report:
        with Path(args.base_report).open() as handle:
            base_report = json.load(handle)
        if base_report.get("collision_scope") != args.collision_scope:
            raise ValueError("--collision-scope does not match the B3a-0b base report")
        if base_report.get("route_end_policy") != args.route_end_policy:
            raise ValueError("--route-end-policy does not match the B3a-0b base report")

    expert_by_key = _load_expert_profiles(Path(args.expert_profiles))
    missing_expert = [str(key) for key in base["key"] if str(key) not in expert_by_key]
    if missing_expert:
        raise KeyError(
            f"missing {len(missing_expert)} expert profiles; first={missing_expert[0]}"
        )
    expert_velocity = np.stack([expert_by_key[str(key)][0] for key in base["key"]])
    cached_current = np.asarray([expert_by_key[str(key)][1] for key in base["key"]])
    current_speed = np.maximum(base["current_speed"], 0.0)
    if not np.allclose(cached_current, current_speed, atol=1e-3):
        raise ValueError("expert-profile and B3a-0b current speeds do not match")

    shards, actor_location = _load_actor_cache(Path(args.future_cache_dir))
    missing_actors = [str(key) for key in base["key"] if str(key) not in actor_location]
    if missing_actors:
        raise KeyError(
            f"missing {len(missing_actors)} future actor frames; first={missing_actors[0]}"
        )

    manual_arrays = {key: value for key, value in base.items()}
    manual_reference: dict[str, Any] = {}
    for margin_index, margin in enumerate(margins):
        manual_reference[f"{float(margin):g}"] = {}
        for scope_name, scope in {
            "all": np.ones(n, dtype=bool),
            "multi": base["multi"],
        }.items():
            if scope.any():
                manual_reference[f"{float(margin):g}"][scope_name] = summarize_margin(
                    manual_arrays, margin_index, scope
                )

    raw_distance = base["candidate_distance"][:, 0]
    raw_velocity = interval_speeds_from_distances(raw_distance, interval_s)
    raw_expert_mae = np.abs(raw_velocity - expert_velocity).mean(axis=1)
    include_class_ids = (1, 2) if args.collision_scope == "dynamic" else None
    extrapolate = args.route_end_policy == "extrapolate"
    scopes_base = {
        "key": base["key"],
        "multi": base["multi"],
        "current_speed": current_speed,
        "expert_brake": base["expert_brake"],
        "expert_velocity": expert_velocity,
    }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "base_frames": str(base_path.resolve()),
        "vocab_dir": str(vocab_dir.resolve()),
        "expert_profiles": str(Path(args.expert_profiles).resolve()),
        "future_cache_dir": str(Path(args.future_cache_dir).resolve()),
        "n_frames": n,
        "future_times_s": times.tolist(),
        "safety_margins_m": margins.tolist(),
        "collision_scope": args.collision_scope,
        "route_end_policy": args.route_end_policy,
        "velocity_representation": args.representation,
        "route_selection": "reused confidence-winner spatial route from B3a-0b; route never changes",
        "oracle_privilege": "GT future actors keep safe raw, otherwise choose maximum-progress safe vocabulary profile",
        "profile_policy": {
            "progress_limited": "vocabulary progress must not exceed raw progress",
            "start_reachable_progress_limited": (
                "progress-limited plus first-interval acceleration within configured bounds"
            ),
            "max_accel_mps2": args.max_accel,
            "max_decel_mps2": args.max_decel,
        },
        "manual_profile_reference": manual_reference,
        "results": {},
    }

    for cluster_count in clusters:
        filename = (
            f"velocity_vocab_k{cluster_count}.npy"
            if args.representation == "absolute"
            else f"relative_velocity_vocab_k{cluster_count}.npy"
        )
        vocab_path = vocab_dir / filename
        vocabulary = np.load(vocab_path).astype(np.float32)
        if vocabulary.shape != (cluster_count, len(times)):
            raise ValueError(
                f"{vocab_path} has shape {vocabulary.shape}; expected {(cluster_count, len(times))}"
            )
        if args.representation == "absolute":
            realized_velocity = np.broadcast_to(
                vocabulary[None], (n, *vocabulary.shape)
            )
        else:
            realized_velocity = realize_relative_profiles(vocabulary, current_speed)
        distances = profile_distances(realized_velocity, interval_s)
        progress = distances[:, :, -1]
        within_raw = progress <= raw_distance[:, -1, None] + 1e-4
        start_reachable = start_reachability_mask(
            realized_velocity,
            current_speed,
            interval_s,
            max_accel_mps2=args.max_accel,
            max_decel_mps2=args.max_decel,
        )
        transition_reachable = transition_reachability_mask(
            realized_velocity,
            interval_s,
            max_accel_mps2=args.max_accel,
            max_decel_mps2=args.max_decel,
        )
        expert_mae = np.abs(expert_velocity[:, None] - realized_velocity).mean(axis=2)
        collision = np.zeros((n, len(margins), cluster_count), dtype=bool)
        ttc = np.full((n, len(margins), cluster_count), np.inf, dtype=np.float32)
        overflow = distances[:, :, -1] > base["route_length"][:, None] + 1e-4

        iterator = zip(base["key"], base["route"], strict=True)
        for row, (key, route) in enumerate(
            tqdm(iterator, total=n, desc=f"B3a-1 vocabulary K={cluster_count}")
        ):
            shard_path, local_index = actor_location[str(key)]
            actors = unpack_future_actor_frame(shards[shard_path], local_index)
            for candidate_index, candidate_distance in enumerate(distances[row]):
                positions, yaws = interpolate_route_by_distance(
                    route, candidate_distance, extrapolate=extrapolate
                )
                for margin_index, margin in enumerate(margins):
                    result = future_collision_label(
                        positions,
                        yaws,
                        actors,
                        safety_margin_m=float(margin),
                        include_class_ids=include_class_ids,
                    )
                    collision[row, margin_index, candidate_index] = result.collision
                    ttc[row, margin_index, candidate_index] = result.ttc_s

        terminal_delta = realized_velocity[:, :, -1] - current_speed[:, None]
        family = np.full((n, cluster_count), "steady", dtype="U16")
        family[realized_velocity[:, :, -1] <= 0.3] = "stopped"
        family[(realized_velocity[:, :, -1] > 0.3) & (terminal_delta <= -1.0)] = (
            "braking"
        )
        family[terminal_delta >= 1.0] = "accelerating"
        arrays = {
            **scopes_base,
            "raw_collision": base["candidate_collision"][:, :, 0],
            "raw_distance": raw_distance,
            "raw_velocity": raw_velocity,
            "raw_expert_mae": raw_expert_mae,
            "vocab_collision": collision,
            "vocab_distance": distances,
            "vocab_velocity": realized_velocity,
            "vocab_expert_mae": expert_mae,
            "vocab_overflow": overflow,
            "vocab_family": family,
        }
        policies = {
            "progress_limited": within_raw,
            "start_reachable_progress_limited": within_raw & start_reachable,
        }
        cluster_result: dict[str, Any] = {
            "vocabulary": str(vocab_path.resolve()),
            "transition_reachable_candidate_count": int(transition_reachable.sum()),
            "transition_reachable_candidate_fraction": float(
                transition_reachable.mean()
            ),
            "all_vocab_nearest_expert_mae_mps": {
                "mean": float(expert_mae.min(axis=1).mean()),
                **_quantiles(expert_mae.min(axis=1)),
            },
            "margins": {},
        }
        behavior_scopes = _behavior_scopes(arrays)
        for margin_index, margin in enumerate(margins):
            margin_result: dict[str, Any] = {}
            for policy_name, policy_valid in policies.items():
                margin_result[policy_name] = {
                    scope_name: summarize_policy(
                        arrays, margin_index, scope, policy_valid
                    )
                    for scope_name, scope in behavior_scopes.items()
                    if scope.any()
                }
                margin_result[f"{policy_name}_union_manual"] = {
                    scope_name: summarize_manual_union(
                        base, arrays, margin_index, scope, policy_valid
                    )
                    for scope_name, scope in behavior_scopes.items()
                    if scope.any()
                }
            cluster_result["margins"][f"{float(margin):g}"] = margin_result
        payload["results"][str(cluster_count)] = cluster_result

        frame_path = output.with_name(f"{output.stem}_k{cluster_count}.frames.npz")
        np.savez_compressed(
            frame_path,
            key=base["key"],
            vocabulary=vocabulary,
            realized_velocity=realized_velocity,
            vocabulary_distance=distances,
            raw_collision=base["candidate_collision"][:, :, 0],
            vocabulary_collision=collision,
            vocabulary_ttc=ttc,
            vocabulary_overflow=overflow,
            within_raw_progress=within_raw,
            start_reachable=start_reachable,
            transition_reachable=transition_reachable,
            expert_profile_mae=expert_mae,
            margins_m=margins,
            future_times_s=times,
        )
        cluster_result["frame_records"] = str(frame_path.resolve())

    with output.open("w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nB3a-1 velocity-vocabulary oracle ({n} held-out frames; route fixed)")
    print(f"collision={args.collision_scope}, route-end={args.route_end_policy}")
    for margin in margins:
        margin_key = f"{float(margin):g}"
        manual = manual_reference[margin_key]["all"]
        print(
            f"\nmargin={float(margin):.2f} manual-rescue="
            f"{manual['expanded_rescue_rate_given_raw_collision']:.1%} "
            f"manual-unresolved={manual['all_candidates_unsafe_count']}"
        )
        for cluster_count in clusters:
            result = payload["results"][str(cluster_count)]["margins"][margin_key]
            primary = result["progress_limited"]["all"]
            reachable = result["start_reachable_progress_limited"]["all"]
            union = result["start_reachable_progress_limited_union_manual"]["all"]
            print(
                f"K={cluster_count:2d} progress-limited candidates="
                f"{primary['candidate_count']['mean']:.1f} rescue="
                f"{primary['rescue_rate_given_raw_collision']:.1%} "
                f"unresolved={primary['all_candidates_unsafe_count']} "
                f"oracle-coll={primary['oracle_collision_rate']:.1%} "
                f"moving/stop={primary['moving_rescue_rate_given_raw_collision']:.1%}/"
                f"{primary['stopping_rescue_rate_given_raw_collision']:.1%}"
            )
            print(
                f"     start-reachable candidates={reachable['candidate_count']['mean']:.1f} "
                f"zero={reachable['candidate_count']['zero_candidate_rate']:.1%} "
                f"rescue={reachable['rescue_rate_given_raw_collision']:.1%} "
                f"unresolved={reachable['all_candidates_unsafe_count']}"
            )
            print(
                f"     manual+reachable rescue="
                f"{union['union_rescue_rate_given_raw_collision']:.1%} "
                f"incremental={union['incremental_vocabulary_rescue_count']} "
                f"unresolved={union['all_union_candidates_unsafe_count']}"
            )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
