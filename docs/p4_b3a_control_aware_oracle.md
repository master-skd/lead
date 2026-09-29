# B3a control-aware scalar-action oracle

This is an offline feasibility audit, not a new closed-loop policy. Visual
Intent, route confidence, and the confidence-selected geometric Path stay
frozen. Candidate actions are the scalar target speeds the current
longitudinal controller actually receives: baseline, -0.5, -1, -2 m/s, stop,
and +1 m/s. Duplicate targets are masked. The +1 action is reported for oracle
coverage but is **not** permitted by the baseline-preserving slowdown oracle.

For each action, the audit holds the target for 2 seconds, calls LEAD's actual
`get_throttle` and `brake_ratio` logic at 20 Hz, advances an explicit
acceleration/braking model, and interpolates the traveled distance on the
frozen Path. GT future actor boxes provide SAT collision/TTC labels. Saved
per-action dimensions are collision, TTC, progress, comfort, brake fraction,
terminal speed, and route-end overflow. The GT oracle changes the baseline
only when the baseline trajectory collides and a collision-free slowdown stays
within a specified progress-loss budget.

The controller *decision* is exact, but the acceleration model is not CARLA
physics. The Path is reconstructed from cached candidate states, so positions
beyond its sampled endpoint are extrapolated and marked as unverified. There
is no lateral tracking error, perception error, or future replanning. These
GT-actor labels must never be used as an inference-time selection policy.

Run a distributed sample of held-out frames on CPU (no GPU burn):

```bash
python scripts/p4/audit_b3a_control_aware_actions.py \
  --limit 5000 --sample-per-shard 600 \
  --output outputs/local_training/p5_stepB3a_control_aware_oracle/nominal_5000.json
```

Use `--limit 0 --sample-per-shard 0` for the entire held-out split. The script
saves a JSON summary plus a `.frames.npz` with all action-level labels, suitable
for a later multidimensional scorer without rebuilding the candidate outcomes.
The `--drive-accel`, `--brake-decel`, and `--drag` parameters support sensitivity
checks; defaults are 4.8, 6.0, and 0.2 m/s².

## First 5,000-frame result

The sample spans the nine held-out feature shards. Values below are counts;
`rescue @2m/@4m` only includes the 3,500+ frames where the baseline stays
within the cached Path and also excludes alternative actions that overflow it.

| Dynamics (drive/brake m/s²) | Raw collision / 5000 | Full-sample slowdown rescue @2m | Path-qualified raw collisions | Path-qualified rescue @2m / @4m |
| --- | ---: | ---: | ---: | ---: |
| 3.5 / 4.0 | 146 | 32 | 91 | 15 / 28 |
| 4.8 / 6.0 | 141 | 34 | 90 | 16 / 30 |
| 6.0 / 8.0 | 142 | 25 | 93 | 11 / 30 |

In the nominal run, 1,462/5,000 baseline rollouts overflow the cached route
geometry; do **not** use the full-sample oracle floor as a deployment claim.
For the 3,538 path-qualified frames, the raw collision rate is 90/3,538;
under a 2 m progress-loss limit, the GT oracle can avoid 16 of those 90,
while a 4 m limit permits 30. This is *potential* safety headroom, not evidence
that a learned scorer can identify those frames without false braking.

On the same 5,000 frames, the nominal control rollout's terminal progress
differs from the old candidate-zero velocity profile by more than 1 m in
22.4% of frames, and their raw collision labels disagree in 0.66% of frames.
Thus the old full-profile safety target is not interchangeable with the action
actually supplied to the controller.

Next decision: before training a DrivoR-style scorer on these labels, either
cache the complete selected Path or restrict training/evaluation to verified
geometry, then validate the acceleration model against actual CARLA control
traces. Only if that audit preserves useful collision-free, progress-feasible
alternatives should the multidimensional scorer be trained.
