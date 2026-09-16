#!/usr/bin/env bash
# B3a-0b: expanded velocity profiles and multi-margin GT collision oracle.
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
LEAD_ENV="/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead"
GPU_BURN_DIR="/mmu_mllm_hdd_3/liuzihan08/gpu-burn"
CKPT_DIR="${B3A_CKPT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB2_corridor}"
SPLIT_DIR="${B3A_SPLIT_DIR:-${CKPT_DIR}/b2d_route_split}"
CACHE_DIR="${B3A_CACHE_DIR:-${CKPT_DIR}/b2d_safety_features/heldout_future_actors}"
OUTPUT_DIR="${B3A_OUTPUT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_velocity_oracle}"
LIMIT=5000

args=("$@")
for ((index=0; index<${#args[@]}; index++)); do
    if [[ "${args[$index]}" == "--limit" ]]; then
        if (( index + 1 >= ${#args[@]} )); then
            echo "--limit requires a value" >&2
            exit 2
        fi
        LIMIT="${args[$((index + 1))]}"
    fi
done

if [[ ! -x "${LEAD_ENV}/bin/python" ]]; then
    echo "lead environment not found: ${LEAD_ENV}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"
"${LEAD_ENV}/bin/python" scripts/p4/make_b2d_route_split.py \
    --manifest data/p4/manifest.jsonl \
    --output-dir "${SPLIT_DIR}" \
    --heldout-fraction 0.1 \
    --seed 20260901
"${LEAD_ENV}/bin/python" scripts/p4/build_b2d_future_actor_cache.py \
    --manifest "${SPLIT_DIR}/heldout.jsonl" \
    --cache-dir "${CACHE_DIR}" \
    --limit "${LIMIT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
"${LEAD_ENV}/bin/python" scripts/p4/eval_b3a_velocity_profile_oracle.py \
    --ckpt-dir "${CKPT_DIR}" \
    --heldout-manifest "${SPLIT_DIR}/heldout.jsonl" \
    --future-cache-dir "${CACHE_DIR}" \
    --anchor-source predicted \
    --batch-size 16 \
    --num-workers 4 \
    --limit "${LIMIT}" \
    --safety-margins 0,0.1,0.2 \
    --out "${OUTPUT_DIR}/velocity_profile_oracle.json" \
    "$@"

if [[ "${B3A_SKIP_GPU_BURN:-0}" != "1" ]]; then
    if [[ ! -x "${GPU_BURN_DIR}/gpu_burn" || ! -f "${GPU_BURN_DIR}/compare.ptx" ]]; then
        echo "gpu-burn binary or compare.ptx not found in: ${GPU_BURN_DIR}" >&2
        exit 1
    fi
    cd "${GPU_BURN_DIR}"
    mkdir -p burn_logs
    GPU_BURN_LOG="${GPU_BURN_DIR}/burn_logs/b3a_velocity_profile_$(date -u +%Y%m%dT%H%M%SZ).log"
    nohup ./gpu_burn -m 70% 99999999 >"${GPU_BURN_LOG}" 2>&1 < /dev/null &
    GPU_BURN_PID=$!
    echo "${GPU_BURN_PID}" > "${GPU_BURN_DIR}/burn_logs/b3a_velocity_profile_latest.pid"
    echo "gpu_burn started in background: pid=${GPU_BURN_PID} log=${GPU_BURN_LOG}"
fi
