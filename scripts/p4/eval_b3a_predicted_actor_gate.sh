#!/usr/bin/env bash
# Frozen LEAD detections -> constant-velocity actor futures -> velocity gate audit.
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
LEAD_ENV="/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead"
DENSE_ROOT="${B3A_V2_DENSE_ROOT:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_v2_dense_data}"
FEATURE_DIR="${B3A_V2_FEATURE_ROOT:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_v2_scene_scorer/features}/heldout"
OUTPUT_DIR="${B3A_PRED_ACTOR_OUTPUT:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_predicted_actor_gate}"
CKPT_DIR="${B3A_CKPT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB2_corridor}"
LIMIT="${B3A_PRED_ACTOR_LIMIT:-5000}"

cd "${REPO_ROOT}"
"${LEAD_ENV}/bin/python" scripts/p4/extract_b3a_predicted_actor_boxes.py \
    --ckpt-dir "${CKPT_DIR}" \
    --manifest "${DENSE_ROOT}/route_split/heldout.jsonl" \
    --nearest-vlm-manifest "${REPO_ROOT}/data/p4/manifest.jsonl" \
    --feature-cache-dir "${FEATURE_DIR}" \
    --output-dir "${OUTPUT_DIR}/predicted_boxes" \
    --limit "${LIMIT}" \
    --batch-size "${B3A_PRED_ACTOR_BATCH_SIZE:-16}" \
    --num-workers "${B3A_PRED_ACTOR_WORKERS:-4}"

"${LEAD_ENV}/bin/python" scripts/p4/eval_b3a_predicted_actor_gate.py \
    --feature-cache-dir "${FEATURE_DIR}" \
    --predicted-box-dir "${OUTPUT_DIR}/predicted_boxes" \
    --out "${OUTPUT_DIR}/predicted_actor_gate.json" \
    "$@"
