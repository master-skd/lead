#!/usr/bin/env bash
# Route-held-out B2d K-arm audit. Uses privileged future actors only for labels.
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
LEAD_ENV="/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead"
GPU_BURN_DIR="/mmu_mllm_hdd_3/liuzihan08/gpu-burn"
CKPT_DIR="${B2D_CKPT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB2_corridor}"
SPLIT_DIR="${B2D_SPLIT_DIR:-${CKPT_DIR}/b2d_route_split}"
CACHE_DIR="${B2D_HELDOUT_CACHE_DIR:-${CKPT_DIR}/b2d_heldout_future_actor_cache}"
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

cd "${REPO_ROOT}"
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
"${LEAD_ENV}/bin/python" scripts/p4/eval_b2d_karm_dynamic_safety.py \
    --ckpt-dir "${CKPT_DIR}" \
    --heldout-manifest "${SPLIT_DIR}/heldout.jsonl" \
    --future-cache-dir "${CACHE_DIR}" \
    --anchor-source predicted \
    --batch-size 16 \
    --num-workers 4 \
    --limit "${LIMIT}" \
    --out "${CKPT_DIR}/b2d_karm_dynamic_safety.json"

if [[ "${B2D_SKIP_GPU_BURN:-0}" != "1" ]]; then
    cd "${GPU_BURN_DIR}"
    mkdir -p burn_logs
    GPU_BURN_LOG="${GPU_BURN_DIR}/burn_logs/b2d_karm_$(date -u +%Y%m%dT%H%M%SZ).log"
    nohup ./gpu_burn -m 70% 99999999 >"${GPU_BURN_LOG}" 2>&1 < /dev/null &
    GPU_BURN_PID=$!
    echo "${GPU_BURN_PID}" > "${GPU_BURN_DIR}/burn_logs/b2d_karm_latest.pid"
    echo "gpu_burn started in background: pid=${GPU_BURN_PID} log=${GPU_BURN_LOG}"
fi
