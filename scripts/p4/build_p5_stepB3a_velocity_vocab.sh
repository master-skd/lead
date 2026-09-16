#!/usr/bin/env bash
# Build expert velocity vocabularies; this stage is CPU-only and does not load LEAD.
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
LEAD_ENV="/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead"
GPU_BURN_DIR="/mmu_mllm_hdd_3/liuzihan08/gpu-burn"
SPLIT_DIR="${B3A_SPLIT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB2_corridor/b2d_route_split}"
OUTPUT_DIR="${B3A_VELOCITY_VOCAB_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_velocity_vocab}"

cd "${REPO_ROOT}"
export OMP_NUM_THREADS="${B3A_KMEANS_THREADS:-16}"
"${LEAD_ENV}/bin/python" scripts/p4/build_b3a_velocity_vocab.py \
    --train-manifest "${SPLIT_DIR}/train.jsonl" \
    --heldout-manifest "${SPLIT_DIR}/heldout.jsonl" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"

if [[ "${B3A_SKIP_GPU_BURN:-0}" != "1" ]]; then
    cd "${GPU_BURN_DIR}"
    mkdir -p burn_logs
    GPU_BURN_LOG="${GPU_BURN_DIR}/burn_logs/b3a_velocity_vocab_$(date -u +%Y%m%dT%H%M%SZ).log"
    nohup ./gpu_burn -m 70% 99999999 >"${GPU_BURN_LOG}" 2>&1 < /dev/null &
    GPU_BURN_PID=$!
    echo "${GPU_BURN_PID}" > "${GPU_BURN_DIR}/burn_logs/b3a_velocity_vocab_latest.pid"
    echo "gpu_burn started in background: pid=${GPU_BURN_PID} log=${GPU_BURN_LOG}"
fi
