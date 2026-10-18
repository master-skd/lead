# P6 candidate / scorer / controller diagnostic (four B2D routes)

This is a **read-only trace** of the existing K-route, one-speed policy. It does
not change the checkpoint, candidate scoring, route selection, target speed, or
control. Use a new tag so the generic B2D driver does not skip existing results.
The diagnostic launcher defaults to routes `3904 3737 2881 24340`, which showed
blocked, 4000-tick, and low-scoring-completed outcomes in the first 220-route run.

On the evaluation machine, after syncing this code:

```bash
cd /root/autodl-tmp/lead
export CARLA_ROOT="$PWD/3rd_party/CARLA_0915"
export P6_B2D_LEAD_PYTHON=/root/autodl-tmp/miniconda3/envs/lead/bin/python
export P6_B2D_QWEN_PYTHON=/path/to/the/Qwen-service-environment/bin/python
export P6_B2D_QWEN_MODEL=/path/to/Qwen3-VL-4B-Instruct
CKPT=outputs/local_training/p6_joint_online_full_8gpu/model_best_joint.pth
bash scripts/p4/eval_p6_diagnostic_routes.sh "$CKPT" "0 1 2 3" p6_single_speed_diag_v1
```

Use the Qwen Python/model paths that worked for the full evaluation. The
existing `eval_p6_joint_b2d.sh` preflight intentionally rejects a K-speed
checkpoint; do not use this wrapper for that checkpoint yet. The command starts
one Qwen service and one CARLA per GPU. If fewer GPUs are free, use `"0"`.
Change `P6_B2D_DIAGNOSTIC_ROUTES="3904 3737"` to run a smaller subset. The
default samples each model tick and queries the CARLA driving-lane map every
fifth tick; `P6_B2D_TRACE_EVERY` and `P6_B2D_TRACE_MAP_EVERY` can change those
rates. Full CARLA trace adds some overhead; do not compare wall-clock time with
the original run.

Each route writes
`outputs/local_evaluation_<tag>/<route>/p6_candidate_trace.jsonl`. Each line
includes the K anchor-conditioned paths, eight 0.25-second proxy states per
candidate, candidate speeds, all seven scorer probabilities, utility, selected
arm, navigation target, ego pose/speed, raw and final control, and a sampled
CARLA-map driving-lane fraction. The command prints a summary at the end; it
can be rerun independently:

```bash
"$P6_B2D_LEAD_PYTHON" scripts/p4/summarize_p6_candidate_trace.py \
  outputs/local_evaluation_p6_single_speed_diag_v1
```

Interpretation: a high best-valid map-lane fraction but low selected fraction
is evidence of a *selection* miss on an available on-road candidate; low
fractions for every arm suggest candidate generation/intent issues. A path can
be on a driving lane yet be the wrong turn, oncoming lane, or collide with a
future actor. Navigation angle is only a coarse direction proxy. The 1-second
predicted-vs-actual ego displacement measures execution discrepancy, but the
policy replans each tick, so it is not a pure PID error. Persistent low target
speed and commanded braking while stationary point to longitudinal control;
normal target speed with stationary ego calls for closer post-processing and
simulator inspection. These traces are diagnostic, not an oracle B2D score.

The generic LEAD evaluator previously passed the literal string `--record None`,
which Bench2Drive treated as a truthy recording directory and called
`start_recorder()`. This revision omits the option, leaving the evaluator's
empty default and avoiding that unrelated recorder request.
