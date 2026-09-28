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
