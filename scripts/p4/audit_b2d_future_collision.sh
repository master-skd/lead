#!/usr/bin/env bash
# B2d data-only audit. No checkpoint or CARLA server is required.
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
LEAD_ENV="/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead"
GPU_BURN_DIR="/mmu_mllm_hdd_3/liuzihan08/gpu-burn"
OUTPUT_DIR="${B2D_OUTPUT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB2_corridor}"
CACHE_DIR="${B2D_CACHE_DIR:-${OUTPUT_DIR}/b2d_future_actor_cache}"
LIMIT=5000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "${LEAD_ENV}/bin/python" ]]; then
    echo "lead environment not found: ${LEAD_ENV}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
"${LEAD_ENV}/bin/python" scripts/p4/build_b2d_future_actor_cache.py \
    --manifest data/p4/manifest.jsonl \
    --cache-dir "${CACHE_DIR}" \
    --limit "${LIMIT}"

"${LEAD_ENV}/bin/python" scripts/p4/audit_b2d_future_collision.py \
    --manifest data/p4/manifest.jsonl \
    --cache-dir "${CACHE_DIR}" \
    --limit "${LIMIT}" \
    --out "${OUTPUT_DIR}/b2d_future_collision_audit.json"

# Match the existing P4 eval scripts: optionally keep every visible GPU occupied.
if [[ "${B2D_SKIP_GPU_BURN:-0}" != "1" ]]; then
    if [[ ! -x "${GPU_BURN_DIR}/gpu_burn" || ! -f "${GPU_BURN_DIR}/compare.ptx" ]]; then
        echo "gpu-burn binary or compare.ptx not found in: ${GPU_BURN_DIR}" >&2
        exit 1
    fi
    cd "${GPU_BURN_DIR}"
    mkdir -p burn_logs
    GPU_BURN_LOG="${GPU_BURN_DIR}/burn_logs/b2d_audit_$(date -u +%Y%m%dT%H%M%SZ).log"
    nohup ./gpu_burn -m 70% 99999999 >"${GPU_BURN_LOG}" 2>&1 < /dev/null &
    GPU_BURN_PID=$!
    echo "${GPU_BURN_PID}" > "${GPU_BURN_DIR}/burn_logs/b2d_audit_latest.pid"
    echo "gpu_burn started in background: pid=${GPU_BURN_PID} log=${GPU_BURN_LOG}"
fi
