"""Open-loop validation for B2c': collision risk slows without changing route branch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def _gate_pairs(value: str) -> list[tuple[float, float]]:
    pairs = []
    for item in value.split(","):
        low, high = (float(x) for x in item.split(":"))
        if high <= low:
            raise ValueError(f"invalid gate pair {item!r}: high must exceed low")
        pairs.append((low, high))
    return pairs


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> float:
    return float(num / den.clamp(min=1))


def _binary_auroc(score: torch.Tensor, label: torch.Tensor) -> float:
    """Exact rank-based AUROC without adding a sklearn/torchmetrics dependency."""
    label = label.bool()
    n_pos, n_neg = label.sum(), (~label).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = score.argsort(descending=True)
    y = label[order].float()
    tpr = torch.cat([torch.zeros(1), y.cumsum(0) / n_pos])
    fpr = torch.cat([torch.zeros(1), (~label[order]).float().cumsum(0) / n_neg])
    return float(torch.trapz(tpr, fpr))


def _scope_metrics(
    pred_risk,
    gt_risk,
    raw_speed,
    expert_speed,
    expert_brake,
    route_ade,
    pairs,
    gt_threshold,
):
    from lead.tfv6.route_speed_gate import collision_risk_to_speed_factor

    unsafe = gt_risk >= gt_threshold
    model_brake = raw_speed <= 0.01
    missed_expert_brake = expert_brake & ~model_brake
    quantiles = torch.tensor([0.5, 0.9, 0.95, 0.99])
    common = {
        "n": int(pred_risk.numel()),
        "pred_risk_mean": float(pred_risk.mean()),
        "gt_risk_mean": float(gt_risk.mean()),
        "risk_mae": float((pred_risk - gt_risk).abs().mean()),
        "gt_unsafe_rate": float(unsafe.float().mean()),
        "risk_auroc": _binary_auroc(pred_risk, unsafe),
        "route_ade": float(route_ade.mean()),
        "raw_target_speed": float(raw_speed.mean()),
        "expert_target_speed": float(expert_speed.mean()),
        "raw_target_speed_mae": float((raw_speed - expert_speed).abs().mean()),
        "expert_brake_rate": float(expert_brake.float().mean()),
        "model_brake_rate": float(model_brake.float().mean()),
        "missed_expert_brake_rate": _safe_div(
            missed_expert_brake.sum(), expert_brake.sum(),
        ),
        "pred_risk_quantiles": {
            name: float(value)
            for name, value in zip(
                ("p50", "p90", "p95", "p99"),
                torch.quantile(pred_risk, quantiles),
                strict=True,
            )
        },
    }
    if pred_risk.numel() > 1 and pred_risk.std() > 0 and gt_risk.std() > 0:
        common["risk_correlation"] = float(torch.corrcoef(torch.stack([pred_risk, gt_risk]))[0, 1])
    else:
        common["risk_correlation"] = float("nan")

    gates = {}
    moving = ~model_brake
    safe = ~unsafe
    expert_go = ~expert_brake & moving
    overspeed = moving & (raw_speed > expert_speed + 0.5)
    for low, high in pairs:
        factor = collision_risk_to_speed_factor(pred_risk, low, high)
        trigger = (factor < 0.999) & moving
        stop = (factor < 0.001) & moving
        gated = raw_speed * factor
        true_trigger = trigger & unsafe
        gated_mae = (gated - expert_speed).abs().mean()
        gates[f"{low:g},{high:g}"] = {
            "mean_factor": float(factor.mean()),
            "mean_gated_speed": float(gated.mean()),
            "slow_rate": float(trigger.float().mean()),
            "stop_rate": float(stop.float().mean()),
            "unsafe_recall": _safe_div(true_trigger.sum(), (unsafe & moving).sum()),
            "slow_precision": _safe_div(true_trigger.sum(), trigger.sum()),
            "false_slow_rate_on_safe": _safe_div((trigger & safe).sum(), (safe & moving).sum()),
            # Incremental behaviour relative to the model's original speed prediction.
            # Frames already predicted as brake are excluded: the gate changed nothing there.
            "missed_expert_brake_slow_recovery": _safe_div(
                (trigger & missed_expert_brake).sum(), missed_expert_brake.sum(),
            ),
            "missed_expert_brake_stop_recovery": _safe_div(
                (stop & missed_expert_brake).sum(), missed_expert_brake.sum(),
            ),
            "false_slow_rate_on_expert_go": _safe_div(
                (trigger & expert_go).sum(), expert_go.sum(),
            ),
            "false_stop_rate_on_expert_go": _safe_div(
                (stop & expert_go).sum(), expert_go.sum(),
            ),
            "overspeed_correction_rate": _safe_div(
                (trigger & overspeed).sum(), overspeed.sum(),
            ),
            "target_speed_mae": float(gated_mae),
            "target_speed_mae_delta": float(
                gated_mae - (raw_speed - expert_speed).abs().mean()
            ),
        }
    return {"common": common, "gates": gates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--ckpt-name", default="model_0019.pth")
    ap.add_argument("--anchor-source", choices=("predicted", "cached"), default="predicted")
    ap.add_argument("--gate-pairs", default="0.3:0.7,0.5:0.9,0.7:0.95")
    ap.add_argument("--gt-unsafe-threshold", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.tfv6.collision_cost import collision_cost_per_point
    from lead.tfv6.tfv6 import TFv6
    from lead.training.config_training import TrainingConfig
    from lead.training.mixed_training_utils import mixed_data_collate_fn

    ckpt = os.path.join(args.ckpt_dir, args.ckpt_name)
    out_path = args.out or os.path.join(args.ckpt_dir, "b2c_prime_speed_gate.json")
    with open(os.path.join(args.ckpt_dir, "config.json")) as f:
        config = TrainingConfig(json.load(f), raise_error_on_missing_key=False)
    config.route_selection_mode = "confidence"
    config.route_speed_safety_gate = True

    import timm
    create_model = timm.create_model
    timm.create_model = lambda *a, **k: create_model(*a, **{**k, "pretrained": False})
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        config.use_mixed_precision_training = False
    model = TFv6(device, config).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=True)
    model.eval().requires_grad_(False)

    base = CARLAData(root=config.carla_data, config=config)
    dataset = VLMIntentDataset(
        base, vlm_cache_dir=config.vlm_cache_dir, manifest_path=config.vlm_manifest,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=mixed_data_collate_fn, pin_memory=True,
    )
    near = max(1, min(int(config.route_speed_gate_near_points), config.num_route_points_prediction))
    saved = {
        key: []
        for key in (
            "pred_risk", "gt_risk", "raw_speed", "expert_speed",
            "expert_brake", "route_ade", "multi",
        )
    }
    n_seen = 0

    for data in tqdm(loader, desc=f"B2c' ({args.anchor_source} anchors)"):
        if args.limit is not None and n_seen >= args.limit:
            break
        if args.anchor_source == "predicted":
            data.pop("anchor", None)
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type,
            dtype=config.torch_float_type,
            enabled=config.use_mixed_precision_training and device.type == "cuda",
        ):
            pred = model(data)
            route = pred.pred_route.float()
            gt_point_cost = collision_cost_per_point(
                route.unsqueeze(1),
                data["bev_semantic"].to(route.device),
                config,
                sigma_m=float(config.collision_sigma_m),
            )[:, 0]
        pred_risk = pred.pred_route_collision_risk.float()
        gt_risk = gt_point_cost[:, :near].amax(dim=1).float()
        raw_speed = pred.pred_target_speed_scalar.float().reshape(-1)
        brake_prob = pred.pred_target_speed_distribution.float().softmax(dim=1)[:, 0]
        raw_speed = torch.where(brake_prob > 0.9, torch.zeros_like(raw_speed), raw_speed)
        expert_brake = data["brake"].to(route.device).bool().reshape(-1)
        expert_speed = data["target_speed"].to(route.device).float().reshape(-1)
        expert_speed = torch.where(
            expert_brake, torch.zeros_like(expert_speed), expert_speed,
        )
        label = data["route"].to(route.device).float()
        route_ade = torch.linalg.norm(route - label, dim=-1).mean(dim=1)
        anchors = pred.pred_route_anchor
        valid = (
            anchors[..., 3].to(route.device) > 0.5
            if anchors is not None
            else torch.ones(route.shape[0], 1, device=route.device, dtype=torch.bool)
        )
        bs = route.shape[0]
        keep = bs if args.limit is None else min(bs, args.limit - n_seen)
        values = {
            "pred_risk": pred_risk,
            "gt_risk": gt_risk,
            "raw_speed": raw_speed,
            "expert_speed": expert_speed,
            "expert_brake": expert_brake,
            "route_ade": route_ade,
            "multi": valid.sum(dim=1) > 1,
        }
        for key, value in values.items():
            saved[key].append(value[:keep].detach().cpu())
        n_seen += keep

    values = {key: torch.cat(parts) for key, parts in saved.items()}
    pairs = _gate_pairs(args.gate_pairs)
    all_mask = torch.ones(n_seen, dtype=torch.bool)
    scopes = {}
    for name, mask in (("all", all_mask), ("multi", values["multi"].bool())):
        scopes[name] = _scope_metrics(
            values["pred_risk"][mask], values["gt_risk"][mask],
            values["raw_speed"][mask], values["expert_speed"][mask],
            values["expert_brake"][mask].bool(), values["route_ade"][mask],
            pairs, args.gt_unsafe_threshold,
        )
    payload = {
        "checkpoint": ckpt,
        "anchor_source": args.anchor_source,
        "near_points": near,
        "gt_unsafe_threshold": args.gt_unsafe_threshold,
        "scopes": scopes,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nB2c' speed gate ({n_seen} frames; route is never changed)")
    for scope in ("all", "multi"):
        result = scopes[scope]
        c = result["common"]
        print(f"\n[{scope}] n={c['n']} ADE={c['route_ade']:.3f} risk AUROC={c['risk_auroc']:.3f} "
              f"corr={c['risk_correlation']:.3f} GT-unsafe={c['gt_unsafe_rate']*100:.1f}%")
        print(f"  expert brake={c['expert_brake_rate']*100:.1f}% model brake={c['model_brake_rate']*100:.1f}% "
              f"missed expert brake={c['missed_expert_brake_rate']*100:.1f}% "
              f"speed MAE={c['raw_target_speed_mae']:.3f} m/s")
        print(" low high | slow stop | risk recall/prec/false | brake slow/stop rescue "
              "false-go slow/stop | speed raw->gated  MAE")
        for low, high in pairs:
            g = result["gates"][f"{low:g},{high:g}"]
            print(f" {low:.2f} {high:.2f} | {g['slow_rate']*100:4.1f}% {g['stop_rate']*100:4.1f}% | "
                  f"{g['unsafe_recall']*100:4.1f}/{g['slow_precision']*100:4.1f}/"
                  f"{g['false_slow_rate_on_safe']*100:4.1f}% | "
                  f"{g['missed_expert_brake_slow_recovery']*100:4.1f}/"
                  f"{g['missed_expert_brake_stop_recovery']*100:4.1f}% "
                  f"{g['false_slow_rate_on_expert_go']*100:4.1f}/"
                  f"{g['false_stop_rate_on_expert_go']*100:4.1f}% | "
                  f"{c['raw_target_speed']:.2f}->{g['mean_gated_speed']:.2f} "
                  f"MAE={g['target_speed_mae']:.3f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
