"""GT-privileged Path x Velocity oracle on frozen LEAD held-out frames.

This is a candidate-coverage audit, not a deployable policy: future actors and
the expert route are used for labels / the optional task-direction proxy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def local_path_variants(
    winner_routes: np.ndarray, offsets_m: tuple[float, ...], ramp_m: float = 8.0
) -> np.ndarray:
    """Smooth ego-anchored lateral offsets of the selected geometric path."""

    routes = np.asarray(winner_routes, dtype=np.float32)
    if routes.ndim != 3 or routes.shape[-1] != 2 or routes.shape[1] < 2:
        raise ValueError("winner routes must have shape [B,N>=2,2]")
    if ramp_m <= 0:
        raise ValueError("ramp_m must be positive")
    if not offsets_m:
        return np.empty((len(routes), 0, routes.shape[1], 2), dtype=np.float32)
    path = np.concatenate((np.zeros_like(routes[:, :1]), routes), axis=1)
    segments = np.diff(path, axis=1)
    lengths = np.linalg.norm(segments, axis=-1)
    arc = np.cumsum(lengths, axis=1)
    tangent = np.zeros_like(routes)
    tangent[:, 0] = segments[:, 0]
    tangent[:, -1] = segments[:, -1]
    if routes.shape[1] > 2:
        tangent[:, 1:-1] = segments[:, :-2] + segments[:, 1:-1]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-6)
    normal = np.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)
    u = np.clip(arc / float(ramp_m), 0.0, 1.0)
    ramp = u * u * (3.0 - 2.0 * u)
    offsets = np.asarray(offsets_m, dtype=np.float32)
    return routes[:, None] + offsets[None, :, None, None] * ramp[:, None, :, None] * normal[:, None]


def expert_direction_mask(
    routes: np.ndarray, expert_route: np.ndarray, max_error_deg: float
) -> np.ndarray:
    """GT-privileged heading proxy; not a runtime command-compatibility test."""

    candidate = routes[..., -1, :]
    expert = expert_route[..., -1, :]
    dot = np.einsum("bkd,bd->bk", candidate, expert)
    norm = np.linalg.norm(candidate, axis=-1) * np.linalg.norm(expert, axis=-1)[:, None]
    cosine = dot / np.maximum(norm, 1e-6)
    return (norm > 1e-3) & (cosine >= np.cos(np.deg2rad(max_error_deg)))


def choose_oracle(
    raw_collision: bool,
    collision: np.ndarray,
    eligible: np.ndarray,
    progress: np.ndarray,
    winner: int,
    *,
    joint: bool,
) -> tuple[int, int]:
    """Keep a safe raw; otherwise choose the most-progressing safe candidate."""

    if collision.shape != eligible.shape or collision.shape != progress.shape:
        raise ValueError("collision, eligibility, and progress shapes must match")
    if collision.ndim != 2 or not 0 <= winner < len(collision):
        raise ValueError("expected [K,M] candidates and a valid winner index")
    if not raw_collision:
        return winner, 0
    safe = eligible & ~collision
    if not joint:
        safe = safe.copy()
        safe[np.arange(len(safe)) != winner] = False
    safe[winner, 0] = False  # the raw trajectory is known unsafe
    if not safe.any():
        return winner, 0
    flat = int(np.where(safe, progress, -np.inf).argmax())
    return tuple(map(int, np.unravel_index(flat, collision.shape)))


def candidate_eligibility(
    route_valid: np.ndarray,
    direction_ok: np.ndarray,
    corridor_cost: np.ndarray,
    route_length: np.ndarray,
    candidate_valid: np.ndarray,
    candidate_progress: np.ndarray,
    candidate_pid_target: np.ndarray,
    *,
    raw_progress: float,
    raw_target_speed: float,
    max_corridor_cost: float,
    min_progress_ratio: float,
    conservative: bool,
) -> np.ndarray:
    """Apply task/path, physical, progress and optional no-speedup constraints."""

    arm_ok = route_valid & direction_ok & (corridor_cost <= max_corridor_cost)
    speed_ok = candidate_valid & (candidate_progress >= min_progress_ratio * raw_progress)
    eligible = arm_ok[:, None] & speed_ok[None]
    eligible &= candidate_progress[None] <= route_length[:, None] + 1e-3
    if conservative:
        eligible &= candidate_progress[None] <= raw_progress + 0.25
        eligible &= candidate_pid_target[None] <= raw_target_speed + 0.01
    return eligible


def summarize(
    *,
    raw_collision: np.ndarray,
    fixed_rescue: np.ndarray,
    joint_rescue: np.ndarray,
    joint_arm: np.ndarray,
    winner_arm: np.ndarray,
    raw_progress: np.ndarray,
    joint_progress: np.ndarray,
    scope: np.ndarray,
) -> dict:
    unsafe = raw_collision & scope
    rescued = joint_rescue & unsafe
    denominator = max(int(unsafe.sum()), 1)
    retained = joint_progress[rescued] / np.maximum(raw_progress[rescued], 1e-3)
    return {
        "frames": int(scope.sum()),
        "raw_collision_count": int(unsafe.sum()),
        "fixed_path_rescue_count": int((fixed_rescue & unsafe).sum()),
        "joint_rescue_count": int(rescued.sum()),
        "incremental_joint_rescue_count": int(
            (joint_rescue & ~fixed_rescue & unsafe).sum()
        ),
        "joint_unresolved_count": int((unsafe & ~joint_rescue).sum()),
        "fixed_path_rescue_rate_given_raw_collision": float(
            (fixed_rescue & unsafe).sum() / denominator
        ),
        "joint_rescue_rate_given_raw_collision": float(rescued.sum() / denominator),
        "changed_path_on_rescue_count": int(
            (rescued & (joint_arm != winner_arm)).sum()
        ),
        "rescued_progress_ratio_p10": (
            float(np.quantile(retained, 0.1)) if len(retained) else None
        ),
        "rescued_progress_ratio_median": (
            float(np.median(retained)) if len(retained) else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--nearest-vlm-manifest")
    parser.add_argument("--future-cache-dir", required=True)
    parser.add_argument("--velocity-vocab", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--safety-margin", type=float, default=0.2)
    parser.add_argument("--max-expert-heading-error-deg", type=float, default=45.0)
    parser.add_argument("--max-off-corridor-cost", type=float, default=0.1)
    parser.add_argument("--min-progress-ratio", type=float, default=0.5)
    parser.add_argument("--max-accel", type=float, default=1.89)
    parser.add_argument("--max-decel", type=float, default=4.95)
    parser.add_argument(
        "--local-offsets",
        default="",
        help="comma-separated lateral path offsets in metres, e.g. -1.5,-0.75,0.75,1.5",
    )
    parser.add_argument("--variant-ramp-m", type=float, default=8.0)
    args = parser.parse_args()
    if args.start < 0 or args.limit <= 0 or args.batch_size <= 0:
        raise ValueError("start, limit, and batch-size must be positive/in range")
    if not 0 <= args.min_progress_ratio <= 1:
        raise ValueError("min-progress-ratio must be in [0,1]")
    offsets = tuple(float(value) for value in args.local_offsets.split(",") if value)
    if len(set(offsets)) != len(offsets) or any(abs(value) > 2.0 or value == 0 for value in offsets):
        raise ValueError("local offsets must be unique, nonzero, and within +/-2m")

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from scripts.p4.extract_b3a_velocity_features import _collate
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.future_actor_cache import FUTURE_TIMES_S, unpack_future_actor_frame
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.tfv6.collision_cost import corridor_cost_per_point
    from lead.tfv6.future_collision import future_collision_label, interpolate_route_by_distance
    from lead.tfv6.tfv6 import TFv6
    from lead.tfv6.velocity_scorer import build_velocity_candidates
    from lead.training.config_training import TrainingConfig

    ckpt_dir = Path(args.ckpt_dir)
    with (ckpt_dir / "config.json").open() as handle:
        config = TrainingConfig(json.load(handle), raise_error_on_missing_key=False)
    if not config.multimodal_planner:
        raise ValueError("joint oracle needs a multimodal corridor checkpoint")
    config.route_selection_mode = "confidence"
    config.route_speed_safety_gate = False
    config.route_future_safety_gate = False
    config.route_velocity_scorer_gate = False
    config.route_predicted_actor_velocity_gate = False
    config.use_sensor_perburtation = False
    import timm

    original_create = timm.create_model
    timm.create_model = lambda *a, **k: original_create(
        *a, **{**k, "pretrained": False}
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        config.use_mixed_precision_training = False
    model = TFv6(device, config).to(device)
    model.load_state_dict(
        torch.load(ckpt_dir / "model_0019.pth", map_location=device, weights_only=True),
        strict=True,
    )
    model.eval().requires_grad_(False)
    # Keep the checkpoint's full model architecture for strict loading. Only
    # afterwards disable auxiliary labels that dense data may not provide.
    config.defer_vlm_inputs_to_wrapper = True
    config.detect_boxes = False
    config.use_semantic = False
    config._loaded_config["use_depth"] = False
    config.use_bev_semantic = False

    with Path(args.manifest).open() as handle:
        manifest = [json.loads(line) for line in handle if line.strip()]
    key_by_route_frame = {
        (entry["route"], entry["frame"]): entry["key"] for entry in manifest
    }
    if len(key_by_route_frame) != len(manifest):
        raise ValueError("manifest contains duplicate route/frame pairs")

    base = CARLAData(root=config.carla_data, config=config, random=False)
    dataset = VLMIntentDataset(
        base,
        vlm_cache_dir=config.vlm_cache_dir,
        manifest_path=args.manifest,
        anchor_cache_dir=None,
        nearest_cache_manifest_path=args.nearest_vlm_manifest,
    )
    end = min(args.start + args.limit, len(dataset.valid_indices))
    if args.start >= end:
        raise ValueError("empty held-out interval")
    dataset.valid_indices = dataset.valid_indices[args.start:end]
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )
    if args.num_workers:
        loader_kwargs["prefetch_factor"] = 1
    loader = DataLoader(**loader_kwargs)

    vocabulary = np.load(args.velocity_vocab, allow_pickle=False).astype(np.float32)
    if vocabulary.ndim != 2 or vocabulary.shape[1] != len(FUTURE_TIMES_S):
        raise ValueError("velocity vocabulary must have shape [M,8]")
    vocabulary_tensor = torch.from_numpy(vocabulary)
    interval_s = float(FUTURE_TIMES_S[0])

    # Keep only the keys in this interval, but validate uniqueness and completeness.
    actor_shards = {}
    actor_location = {}
    for path in sorted(Path(args.future_cache_dir).glob("future_actors_*.npz")):
        shard = np.load(path, allow_pickle=False)
        actor_shards[path] = shard
        for index, key in enumerate(shard["keys"]):
            if str(key) in actor_location:
                raise ValueError(f"duplicate future-actor key {key}")
            actor_location[str(key)] = (path, index)
    if not actor_location:
        raise FileNotFoundError(f"no future-actor shards in {args.future_cache_dir}")

    original_path_count = None
    records: dict[str, list] = {name: [] for name in (
        "key", "winner", "multi", "raw_collision", "raw_progress", "raw_target_speed", "route_valid",
        "direction_ok", "corridor_cost", "candidate_valid", "candidate_collision",
        "collision_evaluated", "candidate_progress", "candidate_pid_target", "route_length", "route_ade",
    )}
    for data in tqdm(loader, desc=f"B3b joint oracle {args.start}:{end}"):
        data.pop("anchor", None)  # extract live anchors from the predicted intent blob
        with (
            torch.inference_mode(),
            torch.amp.autocast(
                device_type=device.type,
                dtype=config.torch_float_type,
                enabled=config.use_mixed_precision_training and device.type == "cuda",
            ),
        ):
            prediction = model(data)
        original_routes_tensor = prediction.pred_route_multimodal
        anchors_tensor = prediction.pred_route_anchor
        intent_tensor = prediction.pred_visual_intent
        required = (
            original_routes_tensor,
            anchors_tensor,
            intent_tensor,
            prediction.pred_route_selected_idx,
        )
        if any(value is None for value in required):
            raise RuntimeError("model did not expose all route arms, anchors, or intent")
        original_routes = original_routes_tensor.float().cpu().numpy()
        if original_path_count is None:
            original_path_count = original_routes.shape[1]
        elif original_path_count != original_routes.shape[1]:
            raise ValueError("number of original paths changed between batches")
        anchors = anchors_tensor.float().cpu().numpy()
        winner = prediction.pred_route_selected_idx.long().cpu().numpy()
        expert = data["route"].float().cpu().numpy()
        original_valid = anchors[..., 3] > 0.5
        variants = local_path_variants(
            original_routes[np.arange(len(winner)), winner], offsets,
            ramp_m=args.variant_ramp_m,
        )
        routes = np.concatenate((original_routes, variants), axis=1)
        route_valid = np.concatenate((
            original_valid,
            np.broadcast_to(
                original_valid[np.arange(len(winner)), winner, None],
                (len(winner), len(offsets)),
            ),
        ), axis=1)
        routes_tensor = torch.as_tensor(routes, device=original_routes_tensor.device)
        direction_ok = expert_direction_mask(routes, expert, args.max_expert_heading_error_deg)
        corridor = corridor_cost_per_point(
            routes_tensor.float(),
            torch.sigmoid(intent_tensor.float()),
            config,
            reach_m=float(config.route_corridor_reach_m),
        ).mean(dim=-1).float().cpu().numpy()
        route_ade = np.linalg.norm(routes - expert[:, None], axis=-1).mean(axis=-1)
        current = data["speed"].float().reshape(-1).clamp_min(0)
        raw_target = prediction.pred_target_speed_scalar.float().reshape(-1).cpu()
        candidate_tensor, valid_tensor = build_velocity_candidates(
            current,
            raw_target,
            vocabulary_tensor,
            interval_s=interval_s,
            max_accel_mps2=args.max_accel,
            max_decel_mps2=args.max_decel,
            full_profile_reachability=True,
        )
        candidate = candidate_tensor.numpy()
        candidate_valid = valid_tensor.numpy()
        distances = candidate.cumsum(axis=-1) * interval_s
        pid_target = np.maximum(2 * candidate[..., 0] - current.numpy()[:, None], 0)
        keys = [
            key_by_route_frame[(r, f)]
            for r, f in zip(data["route_number"], data["frame_number"], strict=True)
        ]
        for b, key in enumerate(keys):
            if key not in actor_location:
                raise KeyError(f"missing future actors: {key}")
            path, local_index = actor_location[key]
            actors = unpack_future_actor_frame(actor_shards[path], local_index)
            k_count, m_count = routes.shape[1], candidate.shape[1]
            collision = np.zeros((k_count, m_count), dtype=bool)
            evaluated = np.zeros_like(collision)
            win = int(winner[b])

            def check_candidate(arm: int, mode: int) -> bool:
                positions, yaws = interpolate_route_by_distance(
                    routes[b, arm], distances[b, mode], extrapolate=True
                )
                evaluated[arm, mode] = True
                return future_collision_label(
                    positions,
                    yaws,
                    actors,
                    safety_margin_m=args.safety_margin,
                    include_class_ids=(1, 2),
                ).collision

            collision[win, 0] = check_candidate(win, 0)
            if collision[win, 0]:
                arms_to_label = route_valid[b].copy()
                arms_to_label[win] = True
                for arm in np.flatnonzero(arms_to_label):
                    for mode in np.flatnonzero(candidate_valid[b]):
                        if arm == win and mode == 0:
                            continue
                        collision[arm, mode] = check_candidate(int(arm), int(mode))
            records["key"].append(key)
            records["winner"].append(win)
            records["multi"].append(bool(original_valid[b].sum() > 1))
            records["raw_collision"].append(bool(collision[win, 0]))
            records["raw_progress"].append(float(distances[b, 0, -1]))
            records["raw_target_speed"].append(float(raw_target[b]))
            records["route_valid"].append(route_valid[b])
            records["direction_ok"].append(direction_ok[b])
            records["corridor_cost"].append(corridor[b])
            records["candidate_valid"].append(candidate_valid[b])
            records["candidate_collision"].append(collision)
            records["collision_evaluated"].append(evaluated)
            records["candidate_progress"].append(distances[b, :, -1])
            records["candidate_pid_target"].append(pid_target[b])
            records["route_length"].append(
                np.linalg.norm(np.diff(
                    np.concatenate((np.zeros((k_count, 1, 2)), routes[b]), axis=1),
                    axis=1,
                ), axis=-1).sum(axis=-1)
            )
            records["route_ade"].append(route_ade[b])

    arrays = {name: np.asarray(value) for name, value in records.items()}
    n, k_count, m_count = arrays["candidate_collision"].shape
    selected: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}
    for policy_name, conservative in (("reachable", False), ("conservative", True)):
        fixed_rescue = np.zeros(n, dtype=bool)
        joint_rescue = np.zeros(n, dtype=bool)
        expanded_rescue = np.zeros(n, dtype=bool)
        joint_arm = arrays["winner"].copy()
        joint_mode = np.zeros(n, dtype=np.int16)
        joint_progress = arrays["raw_progress"].copy()
        expanded_arm = arrays["winner"].copy()
        expanded_mode = np.zeros(n, dtype=np.int16)
        expanded_progress = arrays["raw_progress"].copy()
        for row in range(n):
            win = int(arrays["winner"][row])
            eligible = candidate_eligibility(
                arrays["route_valid"][row],
                arrays["direction_ok"][row],
                arrays["corridor_cost"][row],
                arrays["route_length"][row],
                arrays["candidate_valid"][row],
                arrays["candidate_progress"][row],
                arrays["candidate_pid_target"][row],
                raw_progress=float(arrays["raw_progress"][row]),
                raw_target_speed=float(arrays["raw_target_speed"][row]),
                max_corridor_cost=args.max_off_corridor_cost,
                min_progress_ratio=args.min_progress_ratio,
                conservative=conservative,
            )
            progress = np.broadcast_to(
                arrays["candidate_progress"][row][None], (k_count, m_count)
            )
            fixed = choose_oracle(
                bool(arrays["raw_collision"][row]),
                arrays["candidate_collision"][row], eligible, progress, win, joint=False,
            )
            original_eligible = eligible.copy()
            original_eligible[original_path_count:] = False
            joint = choose_oracle(
                bool(arrays["raw_collision"][row]),
                arrays["candidate_collision"][row], original_eligible, progress,
                win, joint=True,
            )
            expanded = choose_oracle(
                bool(arrays["raw_collision"][row]),
                arrays["candidate_collision"][row], eligible, progress,
                win, joint=True,
            )
            fixed_rescue[row] = fixed != (win, 0)
            joint_rescue[row] = joint != (win, 0)
            expanded_rescue[row] = expanded != (win, 0)
            joint_arm[row], joint_mode[row] = joint
            expanded_arm[row], expanded_mode[row] = expanded
            if joint_rescue[row]:
                joint_progress[row] = progress[joint]
            if expanded_rescue[row]:
                expanded_progress[row] = progress[expanded]
        selected[f"{policy_name}_fixed_rescue"] = fixed_rescue
        selected[f"{policy_name}_joint_rescue"] = joint_rescue
        selected[f"{policy_name}_joint_arm"] = joint_arm
        selected[f"{policy_name}_joint_mode"] = joint_mode
        selected[f"{policy_name}_expanded_rescue"] = expanded_rescue
        selected[f"{policy_name}_expanded_arm"] = expanded_arm
        selected[f"{policy_name}_expanded_mode"] = expanded_mode
        summary[policy_name] = {
            scope_name: summarize(
                raw_collision=arrays["raw_collision"],
                fixed_rescue=fixed_rescue,
                joint_rescue=joint_rescue,
                joint_arm=joint_arm,
                winner_arm=arrays["winner"],
                raw_progress=arrays["raw_progress"],
                joint_progress=joint_progress,
                scope=scope,
            )
            for scope_name, scope in (
                ("all", np.ones(n, dtype=bool)), ("multi", arrays["multi"]),
            )
        }
        if offsets:
            for scope_name, scope in (
                ("all", np.ones(n, dtype=bool)), ("multi", arrays["multi"]),
            ):
                expanded_summary = summarize(
                    raw_collision=arrays["raw_collision"],
                    fixed_rescue=fixed_rescue,
                    joint_rescue=expanded_rescue,
                    joint_arm=expanded_arm,
                    winner_arm=arrays["winner"],
                    raw_progress=arrays["raw_progress"],
                    joint_progress=expanded_progress,
                    scope=scope,
                )
                expanded_summary["incremental_local_rescue_count"] = int((
                    arrays["raw_collision"] & scope & expanded_rescue & ~joint_rescue
                ).sum())
                expanded_summary["local_path_selected_on_rescue_count"] = int((
                    arrays["raw_collision"] & scope & expanded_rescue
                    & (expanded_arm >= original_path_count)
                ).sum())
                summary[policy_name][f"expanded_{scope_name}"] = expanded_summary

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_path = output.with_suffix(".frames.npz")
    np.savez_compressed(frame_path, **arrays, **selected)
    report = {
        "start": args.start,
        "end": end,
        "n_frames": n,
        "candidate_shape": [k_count, m_count],
        "original_path_count": original_path_count,
        "local_offsets_m": list(offsets),
        "variant_ramp_m": args.variant_ramp_m,
        "checkpoint": str((ckpt_dir / "model_0019.pth").resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "vocabulary": str(Path(args.velocity_vocab).resolve()),
        "future_cache_dir": str(Path(args.future_cache_dir).resolve()),
        "selection_privilege": (
            "GT future actors; expert endpoint heading is a task-direction proxy, "
            "not a deployable navigation filter"
        ),
        "corridor_source": "predicted Visual Intent blob",
        "policy": {
            "safety_margin_m": args.safety_margin,
            "max_expert_heading_error_deg": args.max_expert_heading_error_deg,
            "max_off_corridor_cost": args.max_off_corridor_cost,
            "min_progress_ratio": args.min_progress_ratio,
            "route_end_policy": "exclude candidates beyond the predicted path length",
            "conservative": (
                "add <= raw 2s distance +0.25m and <= raw first-interval "
                "PID target +0.01m/s"
            ),
            "raw_safe_behavior": "always keep raw candidate",
            "collision_label_scope": (
                "raw candidate on every frame; all physically reachable valid-path "
                "candidates only when raw collides"
            ),
        },
        "diagnostics": {
            "winner_direction_proxy_pass_rate": float(
                arrays["direction_ok"][np.arange(n), arrays["winner"]].mean()
            ),
            "winner_corridor_pass_rate": float(
                (arrays["corridor_cost"][np.arange(n), arrays["winner"]]
                 <= args.max_off_corridor_cost).mean()
            ),
            "mean_valid_arms": float(arrays["route_valid"].sum(axis=1).mean()),
            "mean_original_valid_arms": float(
                arrays["route_valid"][:, :original_path_count].sum(axis=1).mean()
            ),
            "mean_direction_and_corridor_eligible_arms": float((
                arrays["route_valid"]
                & arrays["direction_ok"]
                & (arrays["corridor_cost"] <= args.max_off_corridor_cost)
            ).sum(axis=1).mean()),
            "mean_reachable_vocabulary_profiles_including_raw": float(
                arrays["candidate_valid"].sum(axis=1).mean()
            ),
            "raw_route_overflow_rate": float((
                arrays["raw_progress"]
                > arrays["route_length"][np.arange(n), arrays["winner"]] + 1e-3
            ).mean()),
        },
        "results": summary,
        "frame_records": str(frame_path.resolve()),
    }
    with output.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(f"B3b joint oracle: {n} frames, K={k_count}, M={m_count}")
    for policy, scopes in summary.items():
        for scope, result in scopes.items():
            print(
                f"{policy} [{scope}] raw={result['raw_collision_count']} "
                f"fixed rescue={result['fixed_path_rescue_count']} "
                f"joint rescue={result['joint_rescue_count']} "
                f"incremental={result['incremental_joint_rescue_count']} "
                f"path switches={result['changed_path_on_rescue_count']}"
                + (
                    f" local incremental={result['incremental_local_rescue_count']}"
                    if "incremental_local_rescue_count" in result else ""
                )
            )
    print(f"wrote {output} and {frame_path}")


if __name__ == "__main__":
    main()
