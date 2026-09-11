#!/usr/bin/env bash
# Audit B2d' on the route-held-out expert split, then occupy visible GPUs in background.
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
LEAD_ENV="/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead"
GPU_BURN_DIR="/mmu_mllm_hdd_3/liuzihan08/gpu-burn"
CKPT_DIR="${B2D_CKPT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB2_corridor}"
FEATURE_ROOT="${B2D_FEATURE_ROOT:-${CKPT_DIR}/b2d_safety_features}"
HEAD_DIR="${B2D_HEAD_OUTPUT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB2d_safety_head}"
SPLIT_DIR="${B2D_SPLIT_DIR:-${CKPT_DIR}/b2d_route_split}"

cd "${REPO_ROOT}"
"${LEAD_ENV}/bin/python" scripts/p4/eval_b2d_prime_future_speed_gate.py \
    --feature-cache-dir "${FEATURE_ROOT}/heldout" \
    --heldout-manifest "${SPLIT_DIR}/heldout.jsonl" \
    --head "${HEAD_DIR}/safety_head_best.pth" \
    --out "${HEAD_DIR}/b2d_prime_future_speed_gate.json" \
    "$@"

if [[ "${B2D_SKIP_GPU_BURN:-0}" != "1" ]]; then
    cd "${GPU_BURN_DIR}"
    mkdir -p burn_logs
    GPU_BURN_LOG="${GPU_BURN_DIR}/burn_logs/b2d_prime_$(date -u +%Y%m%dT%H%M%SZ).log"
    nohup ./gpu_burn -m 70% 99999999 >"${GPU_BURN_LOG}" 2>&1 < /dev/null &
    GPU_BURN_PID=$!
    echo "${GPU_BURN_PID}" > "${GPU_BURN_DIR}/burn_logs/b2d_prime_latest.pid"
    echo "gpu_burn started in background: pid=${GPU_BURN_PID} log=${GPU_BURN_LOG}"
fi
