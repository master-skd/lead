# P6: online Qwen intent + joint planner/scorer (first implementation)

Historical configuration: this page describes the initial K-route, shared-speed
run. The later paired-speed variant is documented in
[`p6_multimodal_target_speed.md`](p6_multimodal_target_speed.md). The P7
semantic-intent planner has a different input contract and no scorer; see
[`P7_QwenDrive_AR_Intent_TFv6_Design.md`](../P7_QwenDrive_AR_Intent_TFv6_Design.md).

This run starts from `outputs/local_training/pretrain/model_0030.pth` (encoder and
perception weights, **not** the B2 corridor planner). It freezes the full-data
`vlm_intent_p6_online_full_b32_w4/model_best.pth` decoder. Eight Qwen services,
one per GPU in the `qwenvl` environment, compute 3-camera hidden states directly
from the current RGB batch. The `lead` environment consumes the batch over local
Unix sockets. No `data/p6/vlm_cache_3cam` file is read or generated.

The planner predicts six anchor-conditioned routes and one target speed. Each
route plus that target speed becomes an eight-step, two-second trajectory. The
seven-head cross-attention scorer reads the **current** route trajectories and
planner scene memory; both inputs are detached, so scorer gradients cannot move
the planner. The route head still receives winner route regression, anchor,
collision and corridor losses. `route_conf` and its BCE/padding losses are absent.
Scorer targets are generated from this same batch: GT future dynamic boxes with
SAT collision, lane-graph corridor, navigation route, and expert ego future.

This is a **K-path × one shared planner-speed** experiment, not a full path ×
velocity vocabulary. Dynamic collision supervision covers cars and walkers;
static obstacles remain handled by LEAD's ordinary route collision loss. The
two-second trajectory uses a speed-ramp/interpolated-route proxy, not CARLA PID
rollout. A scorer-selected route must still pass held-out and closed-loop tests
before deployment; the current scorer utility is an uncalibrated diagnostic.

The training manifest restricts examples to the route-separated train split.
Every epoch evaluates a fixed subset of held-out routes and saves
`model_best_joint.pth` by scorer+route validation loss; normal epoch checkpoints
remain available. The original LEAD training dataset (LiDAR, radar, RGB,
semantics, depth, HD-map, bboxes, metadata and bucket collection), the split,
lane-graph labels, Qwen weights, pretrain weights/config and intent checkpoint
must all exist under the repository-relative paths before launching.

Set `P6_JOINT_LEAD_PYTHON`, `P6_JOINT_QWEN_PYTHON` and `P6_JOINT_QWEN_MODEL` to the
target machine's absolute paths, then run `bash scripts/p4/train_p6_joint_online.sh`.
The wrapper accepts `P6_JOINT_BATCH_SIZE`, `P6_JOINT_WORKERS`, `P6_JOINT_EPOCHS`,
`P6_JOINT_VAL_FRAMES`, `P6_JOINT_TRAIN_FRAMES` (smoke cap), and `P6_JOINT_LOGDIR`.
Start with a single-GPU or small
subset smoke test on the target machine; no full-GPU CARLA forward test was
available in the source workspace.

For transfer to a machine that already has the RGB, lane-graph labels, Qwen
model, and route split, run `bash scripts/p4/package_p6_joint_train_assets.sh
--jobs 8`. It creates `outputs/transfer/p6_joint_train_assets/static.tar.gz`
and one `scenario_*.tar.gz` per CARLA scenario. Each archive keeps repository-
relative paths, so extract them at the target repository root. The package
omits `rgb` and all `*_perturbated` sensor folders because this run sets
`use_sensor_perburtation=false`; it retains normal LiDAR, radar, semantic,
depth, HD-map, boxes, metadata, and bucket data. The script writes temporary
files and SHA-256 sidecars, verifies completed archives on rerun, and resumes
by skipping verified shards. Do not transfer `.part` files.

## Training stage timing

The optional `P6_JOINT_QWEN_PREFETCH=1` enables one-batch lookahead for online
Qwen features. After receiving batch *t*'s hidden states, the training rank
submits batch *t+1* to its existing Qwen service before running the planner on
batch *t*. One request at a time uses the same socket and model weights; no
hidden-state files or additional Qwen processes are created. The default is
`0`, so existing launches keep their original synchronous behavior. Because
Qwen and planner share a GPU, concurrent forward work may also cause
contention: compare identical short runs with `0` and `1` on the target
machine before adopting it for a full run. A running process will not switch
modes when this file or the launch script changes; use a new launch/logdir.
With lookahead, `qwen_wait` measures only the residual wait for the prefetched
request and `next_batch_prefetch` includes loading/submitting the following
batch. Compare total step and epoch time, not the Qwen wait alone.

Add `P6_JOINT_TIMING=1 P6_JOINT_TIMING_EVERY=20` to the existing launch
environment. Timing is disabled by default. Each rank writes its own
`$P6_JOINT_LOGDIR/timing/train_rankN.jsonl` and `qwen_rankN.jsonl`; the first
five training steps of each epoch are also recorded. A launch ID separates
restarts, and a request index joins each training step to its Qwen request.
Single-image closed-loop requests and the socket wire protocol are unchanged.

Training stages cover DataLoader waiting, the complete Qwen request, hidden
transfer, model forward, GT labels/loss computation, backward (including DDP
waiting), optimizer/scheduler, and logging. Qwen stages cover payload receive,
PIL conversion, preprocessing, input transfer, model forward, feature extraction
and CPU transfer, and reply sending. Qwen service stages are *within* the
training-side Qwen request, so do not add them to the training step total again.
Epoch setup, loader iterator reset, checkpoint save, and epoch barrier are
recorded separately. No additional distributed collectives are introduced.

Sampled GPU phases synchronize at their boundaries by default, giving wall
times that include completed GPU work instead of just CPU launch times. This
can reduce overlap on sampled steps; unsampled steps run without these syncs.
`P6_JOINT_TIMING_SYNC=0` instead measures host wall time, which cannot accurately
attribute asynchronous GPU work to individual phases. Backward/DDP timing
includes waiting for other ranks and is not a pure compute benchmark. Keep
batch, workers, GPU count, and data configuration fixed for a useful comparison.

Summarize after at least 200 steps, or while training continues:

```bash
python scripts/p4/summarize_p6_joint_timing.py \
  outputs/local_training/p6_joint_online_full_8gpu/timing
```

The report defaults to the latest launch, excludes the first ten steps of each
epoch, shows mean/p50/p95/max stage times and per-rank means, and excludes
validation requests from the Qwen breakdown. Use `--skip-first 0` to examine
startup stalls or a smoke run. Raw JSONL also contains phase start offsets and
wall timestamps for comparing rank arrival times. The script only needs Python's
standard library.

Already-running processes need a restart to use the instrumentation. To preserve
training progress, restart after a completed epoch checkpoint with the same launch
configuration plus `P6_JOINT_RESUME=.../model_NNNN.pth`. The wrapper verifies the
optimizer/scheduler/scaler files and enables LEAD's existing resume mechanism;
the next epoch is `NNNN+1`, and `P6_JOINT_EPOCHS` remains the total target epoch
count. Mid-epoch sampler/worker state is not restored. This option does not
automatically stop or modify any running job.
