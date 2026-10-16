# P6 K-route / one-speed Bench2Drive evaluation

Use the exact P6 epoch checkpoint or `model_best_joint.pth`; the launcher reads
its sibling `config.json` and rejects a K-speed checkpoint. The scorer chooses
one of K valid routes in `TFv6.forward`, while the original single target speed
and selected route are passed to LEAD's existing route/target-speed controller.
The seven-head utility is the current uncalibrated experimental rule, not a
Bench2Drive score or a safety guarantee.

The launcher creates a small evaluation directory containing symlinks to
`config.json` and **one** `model.pth`. This matters because the generic inference
loader otherwise loads every `model*.pth` in a training directory as an ensemble.
It starts one persistent Qwen service per evaluation GPU, using the same
`3cam_drivable` prompt and full three-camera RGB strip as training. It gives
each agent its own socket and correctly passes the physical GPU ID through the
generic Bench2Drive driver. It does not alter checkpoint weights or training
processes. The frozen intent-head checkpoint referenced by `config.json` must
remain available, because model construction loads it before model weights.

On the evaluation machine (after syncing code and weights):

```bash
cd /mmu_nlp_hdd/liuzihan08/lead
export CARLA_ROOT=/path/to/CARLA_0915
export P6_B2D_LEAD_PYTHON=/mmu_nlp_hdd/liuzihan08/miniconda3/envs/lead/bin/python
export P6_B2D_QWEN_PYTHON=/mmu_nlp_hdd/liuzihan08/miniconda3/envs/vllm_qwen35_cu129/bin/python
export P6_B2D_QWEN_MODEL=/mmu_nlp_hdd/liuzihan08/models/Qwen3-VL-4B-Instruct
export ROUTE_TIMEOUT=3600
CKPT=outputs/local_training/REPLACE_WITH_YOUR_P6_RUN/model_0019.pth
P6_B2D_PREFLIGHT_ONLY=1 bash scripts/p4/eval_p6_joint_b2d.sh "$CKPT" "0" p6_kroute_1speed_smoke
B2D_ROUTE_IDS=23687 bash scripts/p4/eval_p6_joint_b2d.sh "$CKPT" "0" p6_kroute_1speed_smoke
```

For all 220 routes, use a new tag and the available GPUs:

```bash
bash scripts/p4/eval_p6_joint_b2d.sh "$CKPT" "0 1 2 3 4 5 6 7" p6_kroute_1speed_full
```

The same tag can be rerun to fill missing routes, but it cannot be reused with
a different checkpoint. Qwen logs are in `outputs/p6_b2d_eval_logs/<tag>/`,
CARLA/route logs in `/tmp/b2d_<tag>/`, individual results in
`outputs/local_evaluation_<tag>/`, and the full-run merge in
`outputs/b2d_<tag>/`. A subset run prints its per-route result count but does
not run the 220-route merge. Do not start this evaluation on GPUs still occupied
by the training run. A preflight alone does not validate closed-loop quality;
inspect the single-route result before committing all 220 routes.
