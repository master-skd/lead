# B3a-v2 scene-conditioned trajectory scorer

This stage keeps Visual Intent and the confidence-selected spatial path frozen. It
only replaces the B3a-v1 velocity MLP with a scorer that consumes complete
`Path x Velocity` trajectories and the planner's scene memory.

## Candidate and scene representation

- Candidate state at each of eight 250 ms intervals:
  `[x, y, sin(yaw), cos(yaw), velocity, acceleration]`.
- Candidate zero remains LEAD's raw target-speed profile.
- K=64 residual velocity profiles are retained only when both the first interval
  and all later interval transitions satisfy acceleration/deceleration limits.
- Scene memory is the frozen planning decoder context: BEV, Visual Intent,
  ego status/navigation command and radar tokens. BEV and intent grids are pooled
  from 10x12 to 4x5; status/radar tokens are preserved.

The scorer uses a temporal Transformer per candidate, followed by a Transformer
decoder that self-attends across candidates and cross-attends to scene memory.

## Independent supervision

Every valid counterfactual candidate receives five soft/binary labels:

- `collision_free`: no SAT box overlap with GT future vehicles/pedestrians.
- `ttc`: 1 for non-collision; otherwise collision time divided by the 2 s horizon.
- `progress`: 70% near-horizon and 30% full-horizon normalized progress.
- `comfort`: exponential penalty from acceleration RMS and jerk RMS.
- `imitation`: `exp(-profile_MAE / 0.5)`, used as a weak prior rather than WTA.

The heads use masked BCE losses. Rare unsafe candidates are reweighted (capped at
20x), and operating thresholds are selected on held-out routes afterward.

At inference, collision-free and TTC first define the feasible set. Progress,
comfort and imitation rank only that set with weights 0.65/0.25/0.10. If it is
empty, the candidate with the best safety score is the explicit fallback.

## Dense expert data

The original cache covers 96,871 of 968,710 frames. Dense extraction uses all
expert frames and exact per-frame RGB/LiDAR/actor/speed data. Missing Qwen-VL
features reuse the nearest cached frame from the same route; extraction reports
mean/p95/max frame gap so this approximation is auditable. Train/held-out splits
remain route-disjoint.

```bash
bash scripts/p4/prepare_p5_stepB3a_v2_dense_data.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  B3A_V2_DENSE=1 \
  bash scripts/p4/build_p5_stepB3a_v2_scene_feature_cache.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  B3A_V2_DENSE=1 \
  bash scripts/p4/train_p5_stepB3a_v2_scene_scorer.sh
```

The trainer saves loss-best and risk-AUROC-best checkpoints for diagnosis.
`selection_status.json` identifies a separate safety-qualified checkpoint only
when held-out selection has no more collisions than the raw profile, rescues at
least as many collisions as it introduces, switches at most 10% of frames,
falls back at most 5%, collision AUROC at least 0.80, within-frame safety
ranking at least 0.60, and profile MAE increase at most 0.10 m/s. The wrapper
evaluates that checkpoint separately when one passes. Do not enable closed-loop
`scene_score` when `safe_checkpoint` is null.
The inference loader checks the checkpoint's `safety_qualified` field, so the
loss-best diagnostic checkpoint cannot be selected accidentally. Set
`route_velocity_scene_scorer_head` to the path in `selection_status.json` after
the safety gate passes.

Use `audit_b3a_trajectory_scene_scorer.py` to inspect each score head's
distribution over valid, raw, selected, collision, and collision-free
candidates. It also reports whether the safety heads rank collision-free
alternatives above colliding candidates within the same frame.

## First 10-epoch run audit (2026-09-23)

The full 96,406-frame held-out threshold sweep gives a raw collision rate of
2.69% and a selected collision rate of 4.27% for the loss-best epoch 4. Every
saved epoch has selected collision rate above the raw profile. The candidate
vocabulary's oracle collision floor is 1.17%, so the selected trajectory is
the limiting factor in this run. No checkpoint passes the deployment gate.

For score distributions, the CPU audit sampled 256 evenly spaced frames per
held-out shard (2,304 frames total). On epoch 4, collision-free scores have
p10/median/p90 = 0.5745/0.5756/0.5762 across valid candidates, with a median
within-frame span of only 0.0026. The 0.5 threshold accepts every raw candidate
in that sample, and the scorer changes speed in 87.5% of sampled frames. The
TTC head puts safe candidates above colliding candidates in only 36.2% of
within-frame safe/unsafe pairs (1,289 pairs across 81 mixed frames), whereas
the collision-free head reaches 65.2%. These pair statistics come from a
sample and should not be mistaken for full held-out measurements.

The current safety labels are heavily imbalanced: about 97% of valid held-out
candidates are collision-free, and the TTC target is also close to 1 for most
candidates. The existing weighted BCE can distinguish some risky scenes by
global AUROC while producing nearly constant scores across alternatives in a
single scene. The next training change should supervise within-scene safe vs.
unsafe ordering and calibrate thresholds before another closed-loop run.
