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
