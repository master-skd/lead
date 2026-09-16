#!/usr/bin/env bash
# Build raw+K64 residual velocity candidate labels on frozen confidence-winner routes.
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
LEAD_ENV="/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead"
CKPT_DIR="${B3A_CKPT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB2_corridor}"
SPLIT_DIR="${B3A_SPLIT_DIR:-${CKPT_DIR}/b2d_route_split}"
ACTOR_ROOT="${B3A_ACTOR_ROOT:-${CKPT_DIR}/b2d_safety_features}"
PROFILE_DIR="${B3A_VELOCITY_VOCAB_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_velocity_vocab}"
RELATIVE_DIR="${B3A_RELATIVE_VOCAB_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_relative_velocity_vocab}"
FEATURE_ROOT="${B3A_VELOCITY_FEATURE_ROOT:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_velocity_scorer/features}"
VOCABULARY="${RELATIVE_DIR}/relative_velocity_vocab_k64.npy"
CHUNK_SIZE="${B3A_VELOCITY_FEATURE_CHUNK_SIZE:-12000}"
EXTRACT_NUM_WORKERS="${B3A_EXTRACT_NUM_WORKERS:-0}"

if [[ ! -x "${LEAD_ENV}/bin/python" ]]; then
    echo "lead environment not found: ${LEAD_ENV}" >&2
    exit 1
fi
if [[ ! -f "${VOCABULARY}" ]]; then
    echo "K=64 relative vocabulary not found: ${VOCABULARY}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
"${LEAD_ENV}/bin/python" scripts/p4/build_b2d_future_actor_cache.py \
    --manifest "${SPLIT_DIR}/train.jsonl" \
    --cache-dir "${ACTOR_ROOT}/train_future_actors"
"${LEAD_ENV}/bin/python" scripts/p4/build_b2d_future_actor_cache.py \
    --manifest "${SPLIT_DIR}/heldout.jsonl" \
    --cache-dir "${ACTOR_ROOT}/heldout_future_actors"

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
    local split="$1"
    local manifest="${SPLIT_DIR}/${split}.jsonl"
    local actor_dir="${ACTOR_ROOT}/${split}_future_actors"
    local profile_cache="${PROFILE_DIR}/${split}_velocity_profiles.npz"
    local output_dir="${FEATURE_ROOT}/${split}"
    local total
    total=$(wc -l < "${manifest}")
    mkdir -p "${output_dir}"
    local starts=()
    for ((start=0; start<total; start+=CHUNK_SIZE)); do starts+=("${start}"); done
    for ((offset=0; offset<${#starts[@]}; offset+=${#DEVICES[@]})); do
        local pids=()
        for ((slot=0; slot<${#DEVICES[@]} && offset+slot<${#starts[@]}; slot++)); do
            local start=${starts[$((offset+slot))]}
            local end=$((start + CHUNK_SIZE))
            if (( end > total )); then end=${total}; fi
            echo "extract velocity ${split} [${start},${end}) on GPU ${DEVICES[$slot]}"
            CUDA_VISIBLE_DEVICES="${DEVICES[$slot]}" "${LEAD_ENV}/bin/python" \
                scripts/p4/extract_b3a_velocity_features.py \
                --ckpt-dir "${CKPT_DIR}" \
                --manifest "${manifest}" \
                --future-cache-dir "${actor_dir}" \
                --expert-profiles "${profile_cache}" \
                --velocity-vocab "${VOCABULARY}" \
                --output-dir "${output_dir}" \
                --start "${start}" --end "${end}" \
                --batch-size 16 --num-workers "${EXTRACT_NUM_WORKERS}" &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do wait "${pid}"; done
    done
}

extract_split train
extract_split heldout

"${LEAD_ENV}/bin/python" - "${FEATURE_ROOT}/train" "${FEATURE_ROOT}/heldout" \
    "$(wc -l < "${SPLIT_DIR}/train.jsonl")" \
    "$(wc -l < "${SPLIT_DIR}/heldout.jsonl")" <<'PY'
import sys
from pathlib import Path
import numpy as np

for directory, expected in ((sys.argv[1], int(sys.argv[3])), (sys.argv[2], int(sys.argv[4]))):
    keys = []
    for path in sorted(Path(directory).glob("velocity_features_*.npz")):
        with np.load(path, allow_pickle=False) as shard:
            keys.extend(shard["keys"].tolist())
    if len(keys) != expected or len(set(keys)) != expected:
        raise SystemExit(
            f"invalid velocity cache {directory}: rows={len(keys)} "
            f"unique={len(set(keys))} expected={expected}"
        )
    print(f"verified {directory}: {expected} unique frames")
PY

echo "B3a velocity feature cache complete: ${FEATURE_ROOT}"
