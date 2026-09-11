#!/usr/bin/env bash
# Train/evaluate the B2d head; the p5_stepB2_corridor checkpoint remains frozen.
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
LEAD_ENV="/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead"
GPU_BURN_DIR="/mmu_mllm_hdd_3/liuzihan08/gpu-burn"
CKPT_DIR="${B2D_CKPT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB2_corridor}"
FEATURE_ROOT="${B2D_FEATURE_ROOT:-${CKPT_DIR}/b2d_safety_features}"
OUTPUT_DIR="${B2D_HEAD_OUTPUT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB2d_safety_head}"
SOURCE_CKPT="${CKPT_DIR}/model_0019.pth"

cd "${REPO_ROOT}"
if ! compgen -G "${FEATURE_ROOT}/train/safety_features_*.npz" >/dev/null || \
   ! compgen -G "${FEATURE_ROOT}/heldout/safety_features_*.npz" >/dev/null; then
    echo "B2d feature cache is missing; first run scripts/p4/build_b2d_safety_feature_cache.sh" >&2
    exit 1
fi

"${LEAD_ENV}/bin/python" scripts/p4/train_b2d_safety_head.py \
    --train-cache-dir "${FEATURE_ROOT}/train" \
    --val-cache-dir "${FEATURE_ROOT}/heldout" \
    --output-dir "${OUTPUT_DIR}" \
    --source-checkpoint "${SOURCE_CKPT}" \
    --epochs 20 --batch-size 4096 --lr 1e-3 --max-pos-weight 30

"${LEAD_ENV}/bin/python" scripts/p4/eval_b2d_safety_head.py \
    --feature-cache-dir "${FEATURE_ROOT}/heldout" \
    --head "${OUTPUT_DIR}/safety_head_best.pth" \
    --out "${OUTPUT_DIR}/heldout_threshold_sweep.json"

if [[ "${B2D_SKIP_GPU_BURN:-0}" != "1" ]]; then
    cd "${GPU_BURN_DIR}"
    mkdir -p burn_logs
    GPU_BURN_LOG="${GPU_BURN_DIR}/burn_logs/b2d_head_$(date -u +%Y%m%dT%H%M%SZ).log"
    nohup ./gpu_burn -m 70% 99999999 >"${GPU_BURN_LOG}" 2>&1 < /dev/null &
    GPU_BURN_PID=$!
    echo "${GPU_BURN_PID}" > "${GPU_BURN_DIR}/burn_logs/b2d_head_latest.pid"
    echo "gpu_burn started in background: pid=${GPU_BURN_PID} log=${GPU_BURN_LOG}"
fi
