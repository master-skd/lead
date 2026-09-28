# B3b-0: joint Path x Velocity coverage oracle

This is an open-loop, **GT-privileged** diagnostic on the held-out route split.
It neither trains nor changes the closed-loop agent. The frozen corridor model
supplies all six predicted Visual Intent paths and its confidence winner. Each
path is composed with the raw target-speed profile and the same current-speed-
relative K16 velocity vocabulary. Complete two-second candidate trajectories
are checked against GT future vehicle and pedestrian boxes.

The comparison is matched within every frame:

1. Raw confidence-winner path and model target speed.
2. Fixed winner path, with any reachable K16 velocity.
3. Any valid Visual Intent path, with any reachable K16 velocity.

Both oracles preserve a safe raw trajectory and otherwise pick the safe,
highest-progress candidate. The report contains a `reachable` policy and a
`conservative` policy, which additionally disallows more two-second distance
or a higher first-interval PID speed target than raw. Candidates must retain
at least 50% of raw two-second progress, stay within the predicted path length,
and pass the predicted Visual Intent corridor cost threshold. Direction is
filtered by a 45-degree comparison to the **GT expert route endpoint**. This
is only a task-direction proxy for the oracle, not an inference-time navigation
mask; its pass rate is reported separately. Treat the result as a candidate
upper bound, not a deployable selection metric.

Run the first 5,000 held-out frames on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/p4/eval_b3b_joint_oracle.sh --limit 5000
```

The default output is
`outputs/local_training/p5_stepB3b_joint_oracle/joint_oracle.json` and a
same-stem `.frames.npz`. Use `--start`/`--limit` with a distinct `B3B_OUTPUT`
name to inspect other non-overlapping intervals. For the complete 96,573-frame
held-out set, set `--limit 96573` after checking the first report; this stage
will take substantially longer because it performs model inference and GT
collision checks for each valid Path x Velocity candidate. No gpu-burn starts
at the end.

The decisive quantity is `incremental_joint_rescue_count`: raw-unsafe frames
where the fixed-path vocabulary has no eligible safe solution but another path
does. Also inspect `changed_path_on_rescue_count`, retained progress and the
winner direction/corridor pass rates before interpreting a low upper bound.

## B3b-1: same-branch local Path variants and denser speed vocabularies

The initial 5,000-frame K16 audit found only 161 frames with more than one
valid original arm, and no incremental Path rescue under the conservative
PID-target constraint. B3b-1 asks whether small detours **within the selected
Path's corridor** help before changing the route generator or training scorer.
Four local variants offset the confidence Path by `-1.5,-0.75,+0.75,+1.5` m,
with a smooth 8 m onset from the ego position. They remain subject to the same
predicted-corridor, expert-direction-proxy, path-length and progress filters.
This is only a geometric upper bound: a lateral-offset curve is not guaranteed
to be dynamically trackable by the existing steering controller.

Run matched 5,000-frame experiments. Use separate output names because the
wrapper's default filename is the original K16 report:

```bash
CUDA_VISIBLE_DEVICES=0 B3B_VOCAB_K=16 \
  B3B_OUTPUT=outputs/local_training/p5_stepB3b_joint_oracle/joint_k16_local.json \
  bash scripts/p4/eval_b3b_joint_oracle.sh --limit 5000 \
    --local-offsets=-1.5,-0.75,0.75,1.5

CUDA_VISIBLE_DEVICES=0 B3B_VOCAB_K=32 \
  B3B_OUTPUT=outputs/local_training/p5_stepB3b_joint_oracle/joint_k32_local.json \
  bash scripts/p4/eval_b3b_joint_oracle.sh --limit 5000 \
    --local-offsets=-1.5,-0.75,0.75,1.5

CUDA_VISIBLE_DEVICES=0 B3B_VOCAB_K=64 \
  B3B_OUTPUT=outputs/local_training/p5_stepB3b_joint_oracle/joint_k64_local.json \
  bash scripts/p4/eval_b3b_joint_oracle.sh --limit 5000 \
    --local-offsets=-1.5,-0.75,0.75,1.5
```

Each report keeps the original `results.<policy>.all` comparison among the
confidence Path and original K arms, and adds `expanded_all` (original K arms
plus local variants). `incremental_local_rescue_count` counts raw-unsafe frames
rescued only by the local variants. Compare the **conservative** policy first,
then check whether reachable-only gains require acceleration beyond the raw
target. Collision labels are exhaustive only on raw-unsafe frames; the
`.frames.npz` file includes `collision_evaluated` to identify tested entries.
Raw-safe frames keep the raw trajectory and need no counterfactual scoring.

For the same local-Path audit on eight GPUs, split the held-out interval into
contiguous shards. The driver gives each worker one GPU, then concatenates the
per-frame NPZ records and recomputes global rates and progress quantiles (it
does not average shard-level percentages). For example:

```bash
B3B_GPU_LIST=0,1,2,3,4,5,6,7 B3B_VOCAB_K=16 \
  B3B_OUTPUT=outputs/local_training/p5_stepB3b_joint_oracle/joint_k16_local_8gpu.json \
  bash scripts/p4/eval_b3b_joint_oracle_8gpu.sh
```

Change `B3B_VOCAB_K` to `32` or `64` and give each run a distinct
`B3B_OUTPUT` filename. `B3B_START`/`B3B_LIMIT` default to `0`/`5000`;
`B3B_LOCAL_OFFSETS` defaults to `-1.5,-0.75,0.75,1.5`. The merged JSON and
same-stem `.frames.npz` appear at `B3B_OUTPUT`; individual shards and logs
are under its `shards/` subdirectory. If any worker fails, there is no merge;
check its log, then rerun. `B3B_CPU_THREADS` defaults to 4 per worker. Each
worker loads its own copy of the model and dataset, so shared-storage
throughput may limit the speedup. No gpu-burn is started.

## B3b-2: nested vocabulary and local-Path feasibility screen

K32 was fitted independently of K16 and is **not** a superset. On the matched
5,000 frames, conservative expanded rescue fell from 91/164 (K16) to 58/164
(K32). The per-frame union of their rescue sets is 95/164, so the additional
coverage from K32 is small. Keep all K16 centers when expanding the vocabulary:

```bash
LEAD_PYTHON=/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead/bin/python
DENSE_ROOT=outputs/local_training/p5_stepB3a_v2_dense_data
NESTED_VOCAB="${DENSE_ROOT}/relative_velocity_vocab/relative_velocity_vocab_k16_plus_k32.npy"
"${LEAD_PYTHON}" scripts/p4/build_b3b_nested_velocity_vocab.py \
  --out "${NESTED_VOCAB}" \
  "${DENSE_ROOT}/relative_velocity_vocab/relative_velocity_vocab_k16.npy" \
  "${DENSE_ROOT}/relative_velocity_vocab/relative_velocity_vocab_k32.npy"

B3B_GPU_LIST=0,1,2,3,4,5,6,7 B3B_VOCAB_PATH="${NESTED_VOCAB}" \
  B3B_OUTPUT=outputs/local_training/p5_stepB3b_joint_oracle/joint_nested_k16_k32_local_8gpu.json \
  bash scripts/p4/eval_b3b_joint_oracle_8gpu.sh
```

The combined vocabulary has 48 residual centers, and the oracle adds its raw
profile for 49 speed candidates per Path. The 8-GPU driver also writes a
same-stem `.feasibility.json`. It screens incremental local-Path rescues for
at least 1 m of two-second progress, prefix curvature, a conservative
`max(speed)^2 * max(curvature)` lateral-acceleration proxy, and per-point
off-corridor cost from **predicted Visual Intent**. Thresholds in this audit
are diagnostic flags, not certified vehicle limits. It does not check true
road occupancy or closed-loop controller tracking. Because the nested file
records its K16 prefix, the audit also reselects the K16-only candidates
from the same forward pass and labels them `*_base_k16`; this avoids a
separate, potentially mismatched baseline run. For moving rescues it also
rechecks the SAT collision after limiting the ego box's yaw change from the
current heading to 30 or 60 degrees/second, while keeping the candidate's
positions unchanged. A collision appearing only after this check suggests
the original oracle relied on an instantaneous heading change; passing it
still does not prove the steering controller can follow the path.

## B3b-3: pose-consistent offline collision recheck

The older oracle interpolates ego positions along the Path and independently
sets the box yaw to the Path tangent. In particular, a local Path offset can
change the ego box's yaw immediately even when it has barely moved. To check
the dependence on this shortcut, `eval_b3b_kinematic_oracle.py` reuses the
saved Path×Velocity lattice and GT actor cache (no model forward or GPU). It
tracks each Path from ego pose `(0,0,0)` with a bounded-yaw unicycle follower,
integrates position and yaw together, and reruns SAT collision. A 2 m
lookahead and 30 degrees/s maximum yaw rate are the default *sensitivity
assumptions*, not measurements of the CARLA controller. It also sweeps
maximum distance to the desired Path at 1 and 2 m.

```bash
/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead/bin/python \
  scripts/p4/eval_b3b_kinematic_oracle.py \
  --report outputs/local_training/p5_stepB3b_joint_oracle/joint_nested_k16_k32_local_8gpu.json \
  --out outputs/local_training/p5_stepB3b_joint_oracle/joint_nested_k16_k32_kinematic_yaw30.json
```

The output reports both the new raw-collision set and the intersection with
the old one (`common_old_and_kinematic_unsafe`). Compare rescues on that
intersection for a paired interpretation; raw collision counts alone can hide
many frame-level changes. Also try `--max-yaw-rate-deg-s 60` and
`--lookahead-m 4` as sensitivity checks. This is still a GT-future oracle and
a simplified follower, not LEAD's lateral PID or a closed-loop score.

On the first 5,000 frames, the original interpolation labels 164 raw collisions.
With 30 degrees/s yaw and 2 m lookahead, the pose-consistent rollout labels
163, but only 133 are the **same frames** (31 old-only, 30 new-only). Under a
1 m maximum tracking-error screen, the conservative oracle rescues 69 with
original Paths and 82 with local variants: 13 incremental local rescues, all
with at least 1 m progress. The incremental count becomes 16 at 60 degrees/s
and 21 at 4 m lookahead. This sensitivity is the reason to check controller
behavior in a small closed-loop diagnostic before treating any oracle count
as an expected driving-score gain.
