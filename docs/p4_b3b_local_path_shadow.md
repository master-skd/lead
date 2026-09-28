# B3b local-Path closed-loop shadow

This runs the frozen B2 corridor checkpoint with its original confidence Path
and original target speed. It checks the raw Path for predicted collision on
every planning tick. Only when that Path is predicted unsafe and ego speed is
at least 1 m/s does it generate four same-branch lateral-offset Paths
(`-1.5,-0.75,+0.75,+1.5` m), compute their predicted-actor collision and
predicted Visual Intent corridor cost, and ask a **copy** of the actual
lateral PID state what steering each Path would produce. The copied
controller never replaces the executed control.

First run one route on the closed-loop machine (with the normal Qwen-VL
service and Bench2Drive prerequisites already running):

```bash
export B2D_ROUTE_IDS="2084"
bash scripts/p4/eval_b3b_local_path_shadow_local.sh \
  outputs/local_training/p5_stepB2_corridor "0" b3b_local_shadow_smoke
```

The route log is
`outputs/local_evaluation_b3b_local_shadow_smoke/2084/local_path_shadow.jsonl`.
Each row includes the executed baseline steer and control, the raw collision
flag, `candidate_evaluated`, and a
`would_switch` flag. That flag requires a predicted raw collision, a
predicted-safe local Path, corridor mean <=0.1, point maximum <=0.25,
current speed >=1 m/s, and a non-saturated PID steer differing by 0.03–0.25
from baseline. It is **not** an executed switch, a learned scorer, or
evidence of true future collision avoidance.
When `candidate_evaluated=true`, the row additionally includes all five
candidate collision flags, corridor costs and counterfactual PID steers.
The predicted actor model uses CenterNet boxes with constant-velocity futures,
and the collision geometry uses the old path-tangent yaw convention; compare
it with the pose-consistent offline audit before promoting it to an active
policy.

If the first route writes the JSONL without errors, run a small matched set:

```bash
export B2D_ROUTE_IDS="2084 2091 2881 3737 24333 2129"
bash scripts/p4/eval_b3b_local_path_shadow_local.sh \
  outputs/local_training/p5_stepB2_corridor "0" b3b_local_shadow_diag6
```

Do not interpret the six-route subset with the 220-route merge score. Review
trigger count, predicted actor count, counterfactual steering saturation and
the route-level infractions first. No GPU burn is started.

## Diagnostic rerun after the six-route shadow audit

The first six-route shadow run had 2,955 ticks, 132 predicted raw-collision
ticks, and 34 ticks with at least one predicted-safe local offset, but zero
`would_switch` decisions. All 50 predicted-safe offset candidates failed the
absolute corridor thresholds; their minimum corridor mean was 0.279. The
executed raw route also often had a high corridor cost. Route 2129's first
actual collision was logged at step 270, before the nearby predicted-risk
episode beginning at step 271. These observations do **not** establish that
the local paths are unsafe or that the corridor transform is wrong: the first
log lacked the predicted intent raster and geometric details needed to tell.

Schema-v2 logs add a compressed uint8 intent-probability map only on expanded
ticks, candidate route and predicted ego poses, per-point corridor costs,
predicted actor futures, and predicted collision TTC/actor ID. The controls
remain exactly the baseline. Rerun just route 2129 first:

```bash
export B2D_ROUTE_IDS="2129"
bash scripts/p4/eval_b3b_local_path_shadow_local.sh \
  outputs/local_training/p5_stepB2_corridor "0" b3b_local_shadow_geometry_2129
```

Summarize it and render a triggered tick (choose a step with
`candidate_evaluated=true` in its JSONL):

```bash
/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead/bin/python \
  scripts/p4/analyze_b3b_local_path_shadow.py \
  outputs/local_evaluation_b3b_local_shadow_geometry_2129
/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead/bin/python \
  scripts/p4/analyze_b3b_local_path_shadow.py \
  outputs/local_evaluation_b3b_local_shadow_geometry_2129/2129 \
  --plot-step 271 --plot-output b3b_2129_step271.png
```

The example step 271 is not guaranteed to trigger on a fresh CARLA rerun.
Use the rerun's actual trigger step. Do not loosen corridor thresholds or
activate Path switching until the raster, coordinates, and TTC have been
checked against the real infraction timing.
