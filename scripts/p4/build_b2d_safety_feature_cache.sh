#!/usr/bin/env bash
# Build frozen predicted-anchor B2d features from the latest corridor checkpoint.
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
LEAD_ENV="/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead"
GPU_BURN_DIR="/mmu_mllm_hdd_3/liuzihan08/gpu-burn"
CKPT_DIR="${B2D_CKPT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB2_corridor}"
SPLIT_DIR="${B2D_SPLIT_DIR:-${CKPT_DIR}/b2d_route_split}"
ROOT_OUT="${B2D_FEATURE_ROOT:-${CKPT_DIR}/b2d_safety_features}"
TRAIN_ACTORS="${ROOT_OUT}/train_future_actors"
VAL_ACTORS="${ROOT_OUT}/heldout_future_actors"
TRAIN_FEATURES="${ROOT_OUT}/train"
VAL_FEATURES="${ROOT_OUT}/heldout"
CHUNK_SIZE="${B2D_FEATURE_CHUNK_SIZE:-12000}"
EXTRACT_NUM_WORKERS="${B2D_EXTRACT_NUM_WORKERS:-0}"

cd "${REPO_ROOT}"

"${LEAD_ENV}/bin/python" scripts/p4/build_b2d_future_actor_cache.py \
    --manifest "${SPLIT_DIR}/train.jsonl" \
    --cache-dir "${TRAIN_ACTORS}"
"${LEAD_ENV}/bin/python" scripts/p4/build_b2d_future_actor_cache.py \
    --manifest "${SPLIT_DIR}/heldout.jsonl" \
    --cache-dir "${VAL_ACTORS}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
else
    GPU_COUNT=$("${LEAD_ENV}/bin/python" -c 'import torch; print(torch.cuda.device_count())')
    if (( GPU_COUNT < 1 )); then
        echo "no CUDA device available" >&2
        exit 1
    fi
    DEVICES=()
    for ((gpu=0; gpu<GPU_COUNT; gpu++)); do DEVICES+=("${gpu}"); done
fi

extract_split() {
    local manifest="$1"
    local actor_dir="$2"
    local feature_dir="$3"
    local total
    total=$(wc -l < "${manifest}")
    mkdir -p "${feature_dir}"
    local starts=()
    for ((start=0; start<total; start+=CHUNK_SIZE)); do starts+=("${start}"); done
    for ((offset=0; offset<${#starts[@]}; offset+=${#DEVICES[@]})); do
        local pids=()
        for ((slot=0; slot<${#DEVICES[@]} && offset+slot<${#starts[@]}; slot++)); do
            local start=${starts[$((offset+slot))]}
            local end=$((start + CHUNK_SIZE))
            if (( end > total )); then end=${total}; fi
            echo "extract ${manifest} [${start},${end}) on GPU ${DEVICES[$slot]}"
            CUDA_VISIBLE_DEVICES="${DEVICES[$slot]}" "${LEAD_ENV}/bin/python" \
                scripts/p4/extract_b2d_safety_features.py \
                --ckpt-dir "${CKPT_DIR}" \
                --manifest "${manifest}" \
                --future-cache-dir "${actor_dir}" \
                --output-dir "${feature_dir}" \
                --start "${start}" --end "${end}" \
                --batch-size 16 --num-workers "${EXTRACT_NUM_WORKERS}" &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do wait "${pid}"; done
    done
}

extract_split "${SPLIT_DIR}/train.jsonl" "${TRAIN_ACTORS}" "${TRAIN_FEATURES}"
extract_split "${SPLIT_DIR}/heldout.jsonl" "${VAL_ACTORS}" "${VAL_FEATURES}"

"${LEAD_ENV}/bin/python" - "${TRAIN_FEATURES}" "${VAL_FEATURES}" \
    "$(wc -l < "${SPLIT_DIR}/train.jsonl")" "$(wc -l < "${SPLIT_DIR}/heldout.jsonl")" <<'PY'
import sys
from pathlib import Path
import numpy as np
for directory, expected in ((sys.argv[1], int(sys.argv[3])), (sys.argv[2], int(sys.argv[4]))):
    keys = []
    for path in sorted(Path(directory).glob("safety_features_*.npz")):
        with np.load(path, allow_pickle=False) as shard:
            keys.extend(shard["keys"].tolist())
    if len(keys) != expected or len(set(keys)) != expected:
        raise SystemExit(f"invalid feature cache {directory}: rows={len(keys)} unique={len(set(keys))} expected={expected}")
    print(f"verified {directory}: {expected} unique frames")
PY

echo "B2d safety feature cache complete: ${ROOT_OUT}"

if [[ "${B2D_SKIP_GPU_BURN:-0}" != "1" ]]; then
    cd "${GPU_BURN_DIR}"
    mkdir -p burn_logs
    GPU_BURN_LOG="${GPU_BURN_DIR}/burn_logs/b2d_features_$(date -u +%Y%m%dT%H%M%SZ).log"
    nohup ./gpu_burn -m 70% 99999999 >"${GPU_BURN_LOG}" 2>&1 < /dev/null &
    GPU_BURN_PID=$!
    echo "${GPU_BURN_PID}" > "${GPU_BURN_DIR}/burn_logs/b2d_features_latest.pid"
    echo "gpu_burn started in background: pid=${GPU_BURN_PID} log=${GPU_BURN_LOG}"
fi
