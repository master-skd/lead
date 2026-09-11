#!/usr/bin/env bash
# B2c' open-loop speed-gate validation. No CARLA/Bench2Drive server is needed.
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
LEAD_ENV="/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead"
GPU_BURN_DIR="/mmu_mllm_hdd_3/liuzihan08/gpu-burn"
CKPT_DIR="${B2C_CKPT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB2_corridor}"

if [[ ! -x "${LEAD_ENV}/bin/python" ]]; then
    echo "lead environment not found: ${LEAD_ENV}" >&2
    exit 1
fi
if [[ ! -f "${CKPT_DIR}/model_0019.pth" ]]; then
    echo "checkpoint not found: ${CKPT_DIR}/model_0019.pth" >&2
    exit 1
fi
if [[ ! -x "${GPU_BURN_DIR}/gpu_burn" || ! -f "${GPU_BURN_DIR}/compare.ptx" ]]; then
    echo "gpu-burn binary or compare.ptx not found in: ${GPU_BURN_DIR}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"${LEAD_ENV}/bin/python" scripts/p4/eval_b2c_prime_speed_gate.py \
    --ckpt-dir "${CKPT_DIR}" \
    --ckpt-name model_0019.pth \
    --anchor-source predicted \
    --batch-size 16 \
    --num-workers 6 \
    "$@"

cd "${GPU_BURN_DIR}"
mkdir -p burn_logs
GPU_BURN_LOG="${GPU_BURN_DIR}/burn_logs/b2c_prime_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup ./gpu_burn -m 70% 99999999 >"${GPU_BURN_LOG}" 2>&1 < /dev/null &
GPU_BURN_PID=$!
echo "${GPU_BURN_PID}" > "${GPU_BURN_DIR}/burn_logs/b2c_prime_latest.pid"
echo "gpu_burn started in background: pid=${GPU_BURN_PID} log=${GPU_BURN_LOG}"
