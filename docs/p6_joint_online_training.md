# P6: online Qwen intent + joint planner/scorer (first implementation)

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
