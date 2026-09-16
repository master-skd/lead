#!/usr/bin/env bash
# Train and evaluate the frozen-route K=64 residual velocity scorer.
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
LEAD_ENV="/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead"
CKPT_DIR="${B3A_CKPT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB2_corridor}"
FEATURE_ROOT="${B3A_VELOCITY_FEATURE_ROOT:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_velocity_scorer/features}"
RELATIVE_DIR="${B3A_RELATIVE_VOCAB_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_relative_velocity_vocab}"
OUTPUT_DIR="${B3A_VELOCITY_SCORER_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_velocity_scorer}"
SOURCE_CKPT="${CKPT_DIR}/model_0019.pth"
VOCABULARY="${RELATIVE_DIR}/relative_velocity_vocab_k64.npy"

if ! compgen -G "${FEATURE_ROOT}/train/velocity_features_*.npz" >/dev/null || \
   ! compgen -G "${FEATURE_ROOT}/heldout/velocity_features_*.npz" >/dev/null; then
    echo "velocity feature cache is missing; run build_p5_stepB3a_velocity_feature_cache.sh first" >&2
    exit 1
fi

cd "${REPO_ROOT}"
"${LEAD_ENV}/bin/python" scripts/p4/train_b3a_velocity_scorer.py \
    --train-cache-dir "${FEATURE_ROOT}/train" \
    --val-cache-dir "${FEATURE_ROOT}/heldout" \
    --output-dir "${OUTPUT_DIR}" \
    --source-checkpoint "${SOURCE_CKPT}" \
    --source-vocabulary "${VOCABULARY}" \
    --epochs 20 --batch-size 256 --lr 1e-3 \
    --collision-loss-weight 1.0 --max-pos-weight 30 \
    "$@"

"${LEAD_ENV}/bin/python" scripts/p4/eval_b3a_velocity_scorer.py \
    --feature-cache-dir "${FEATURE_ROOT}/heldout" \
    --scorer "${OUTPUT_DIR}/velocity_scorer_best.pth" \
    --out "${OUTPUT_DIR}/heldout_selection_sweep.json"
