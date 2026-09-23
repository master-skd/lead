#!/usr/bin/env bash
# Train the scene-conditioned multi-head Path x Velocity scorer.
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
LEAD_ENV="/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead"
CKPT_DIR="${B3A_CKPT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB2_corridor}"
FEATURE_ROOT="${B3A_V2_FEATURE_ROOT:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_v2_scene_scorer/features}"
RELATIVE_DIR="${B3A_RELATIVE_VOCAB_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_relative_velocity_vocab}"
if [[ "${B3A_V2_DENSE:-0}" == "1" ]]; then
    DENSE_ROOT="${B3A_V2_DENSE_ROOT:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_v2_dense_data}"
    RELATIVE_DIR="${B3A_RELATIVE_VOCAB_DIR:-${DENSE_ROOT}/relative_velocity_vocab}"
fi
OUTPUT_DIR="${B3A_V2_SCORER_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_v2_scene_scorer}"
EPOCHS="${B3A_V2_EPOCHS:-10}"
BATCH_SIZE="${B3A_V2_BATCH_SIZE:-1024}"
LR="${B3A_V2_LR:-2e-4}"
DATA_PARALLEL="${B3A_V2_DATA_PARALLEL:-1}"
SOURCE_CKPT="${CKPT_DIR}/model_0019.pth"
VOCABULARY="${RELATIVE_DIR}/relative_velocity_vocab_k64.npy"

PARALLEL_ARGS=()
if [[ "${DATA_PARALLEL}" == "1" ]]; then
    PARALLEL_ARGS+=(--data-parallel)
fi

if ! compgen -G "${FEATURE_ROOT}/train/velocity_features_*.npz" >/dev/null || \
   ! compgen -G "${FEATURE_ROOT}/heldout/velocity_features_*.npz" >/dev/null; then
    echo "B3a-v2 feature cache is missing; build it first" >&2
    exit 1
fi

cd "${REPO_ROOT}"
"${LEAD_ENV}/bin/python" scripts/p4/train_b3a_trajectory_scene_scorer.py \
    --train-cache-dir "${FEATURE_ROOT}/train" \
    --val-cache-dir "${FEATURE_ROOT}/heldout" \
    --output-dir "${OUTPUT_DIR}" \
    --source-checkpoint "${SOURCE_CKPT}" \
    --source-vocabulary "${VOCABULARY}" \
    --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --lr "${LR}" \
    "${PARALLEL_ARGS[@]}" \
    "$@"

"${LEAD_ENV}/bin/python" scripts/p4/eval_b3a_trajectory_scene_scorer.py \
    --feature-cache-dir "${FEATURE_ROOT}/heldout" \
    --scorer "${OUTPUT_DIR}/trajectory_scene_scorer_best.pth" \
    --out "${OUTPUT_DIR}/heldout_threshold_sweep.json"

SAFE_SCORER="$("${LEAD_ENV}/bin/python" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["safe_checkpoint"] or "")' \
    "${OUTPUT_DIR}/selection_status.json")"
if [[ -n "${SAFE_SCORER}" ]]; then
    "${LEAD_ENV}/bin/python" scripts/p4/eval_b3a_trajectory_scene_scorer.py \
        --feature-cache-dir "${FEATURE_ROOT}/heldout" \
        --scorer "${SAFE_SCORER}" \
        --out "${OUTPUT_DIR}/heldout_threshold_sweep_safe.json"
else
    echo "No checkpoint passed the held-out safety gate; do not use the loss-best checkpoint for closed loop." >&2
fi
