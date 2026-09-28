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
