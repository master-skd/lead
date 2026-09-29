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

## Full selected-Path sidecars

The old velocity feature shards contain time-sampled candidate points but not
the complete `pred_route` returned by the frozen corridor model. Points past
their farthest sample cannot be reconstructed. The path-only extraction mode
reruns the **same checkpoint and manifest** and saves `selected_path` as a
separate float32 sidecar per existing shard; it does not rebuild future actors,
velocity candidates, scene tokens, or labels. It needs GPUs for model forward.

On the GPU machine, first run held-out alone for validation, then train:

```bash
B3A_PATH_SPLITS=heldout CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/p4/build_p5_stepB3a_selected_path_cache.sh

B3A_PATH_SPLITS=train CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/p4/build_p5_stepB3a_selected_path_cache.sh
```

The wrapper resumes completed sidecars and verifies every shard against the
existing feature cache: identical keys/order and source checkpoint, matching
selected arm/current speed/raw target, and sampled geometric agreement between
old candidate positions and the newly stored full Path. Verification fails
closed if any source has drifted. No GPU burn is started.

After held-out sidecars pass verification, repeat the oracle on full Paths:

```bash
python scripts/p4/audit_b3a_control_aware_actions.py \
  --selected-paths outputs/local_training/p5_stepB3a_v2_scene_scorer/selected_paths/heldout \
  --limit 5000 --sample-per-shard 600 \
  --output outputs/local_training/p5_stepB3a_control_aware_oracle/full_path_5000.json
```

Then validate the acceleration proxy against CARLA traces before training a
DrivoR-style scorer. Only if the full-Path audit preserves useful
collision-free, progress-feasible alternatives should that training proceed.
