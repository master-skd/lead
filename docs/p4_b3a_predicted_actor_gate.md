# B3a predicted-actor velocity gate audit

This experiment keeps the corridor model's confidence-selected path and the
cached raw + K64 velocity candidates fixed. It uses the frozen checkpoint's
CenterNet detections, not GT actor futures, to decide whether the raw velocity
is risky. Detected car/pedestrian centers are extrapolated for 2 s at the
detected scalar speed along their detected heading. Predicted actor boxes and
ego candidate boxes are compared at the same eight 0.25 s instants with SAT.

The policy retains candidate zero unless its predicted actor collision flag is
true. It then picks the farthest-progressing valid candidate that is predicted
collision-free and does not travel farther than the raw candidate over 2 s.
If no such candidate exists, it retains candidate zero. No learned scorer or
GT future actor state is used to make the decision.

On a GPU machine, from the lead repository root:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/p4/eval_b3a_predicted_actor_gate.sh
```

The default sample is 5,000 evenly spaced held-out feature frames, covering
the existing route-disjoint split. `B3A_PRED_ACTOR_LIMIT` changes the sample
size. Detections are cached in 512-frame shards and completed shards are
skipped on rerun. The summary is written to
`outputs/local_training/p5_stepB3a_predicted_actor_gate/predicted_actor_gate.json`,
with frame-level choices in the companion `.frames.npz` file.

The report compares predicted versus GT collision flags for the raw candidate
and all valid candidates. It also reports GT-actor oracle selection using the
same slowdown-only policy, predicted-gate rescues and introduced collisions,
false switches, speed-profile MAE, and 2 s progress. Detection confidence is
swept at 0.3, 0.5 and 0.7 without rerunning the model.

The actor future is a constant-velocity approximation. It cannot model turns,
braking, interactions or overpass height, and the detected scalar speed is
assumed to follow the detected heading. Thus the audit measures whether this
specific deployable approximation is useful, not an upper bound for all actor
motion predictors. The GT labels come from recorded actor futures and are
used only for evaluation.

## 5,000-frame result and failure taxonomy

With the 0.50 detection threshold, the raw candidate collides in 143/5000
frames (2.86%); predicted-actor selection leaves 117/5000 (2.34%), versus
60/5000 (1.20%) for the GT-actor oracle. It switches 63 frames, rescues 26,
and introduces no new GT collisions in this sample. Mean velocity-profile MAE
changes from 1.08216 to 1.08231 m/s. This is an open-loop result, not a
closed-loop driving-score claim.

To reproduce the error taxonomy from the saved frame choices:

```bash
python scripts/p4/audit_b3a_predicted_actor_gate_cases.py \
  --feature-cache-dir outputs/local_training/p5_stepB3a_v2_scene_scorer/features/heldout \
  --predicted-box-dir outputs/local_training/p5_stepB3a_predicted_actor_gate/predicted_boxes \
  --frame-details outputs/local_training/p5_stepB3a_predicted_actor_gate/predicted_actor_gate.frames.npz \
  --out outputs/local_training/p5_stepB3a_predicted_actor_gate/case_audit_050.json
```

Of 83 GT-rescuable collision frames, the gate rescues 26; 47 are missed
because the raw collision is not predicted, six are flagged but retain raw,
and four switch to another candidate that still collides with a GT actor.
There are 33 switches on GT-safe raw frames, but none introduces a collision
in this sample. The audit JSON contains example frame keys for each group.

## Optional closed-loop A/B

The deployable gate is disabled by default and needs no learned scorer weight.
It uses the confidence-selected spatial path, the dense-data K64 vocabulary
used in this audit, and the existing frozen model's CenterNet detections.
The raw target-speed command is passed through unchanged when the gate does
not switch. On a switch, the first 0.25 s profile interval supplies the PID
target speed; the lateral controller still follows the original path.

On the Bench2Drive machine, update the code and unpack
`p5_b3a_predicted_actor_gate_vocab.tar.gz` from the repository root with
`tar -xzf p5_b3a_predicted_actor_gate_vocab.tar.gz`. It preserves the
repository-relative path
`outputs/local_training/p5_stepB3a_v2_dense_data/relative_velocity_vocab/relative_velocity_vocab_k64.npy`.
Then run the same checkpoint and routes for both arms, using distinct tags:

```bash
bash scripts/p4/eval_b3a_predicted_actor_baseline_local.sh <corridor-checkpoint-dir> "0 1 2 3 4 5 6 7" b3a_actor_baseline
bash scripts/p4/eval_b3a_predicted_actor_gate_local.sh <corridor-checkpoint-dir> "0 1 2 3 4 5 6 7" b3a_actor_gate_050
```

Do not run both arms simultaneously on the same CARLA ports/GPUs. Compare
all 220 matched route IDs, driving score, success rate, collisions, timeouts,
and the gate's `velocity_diagnostics.jsonl` files. Neither the safety head nor
the learned velocity scorer is enabled by the gate wrapper. The gate's
diagnostics are compact: each tick logs predicted raw/selected collision flags,
switch/fallback, selected profile and final control, but not all 65 candidates.
