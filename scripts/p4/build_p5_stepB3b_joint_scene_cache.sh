#!/usr/bin/env bash
# Freeze the B2 corridor checkpoint and cache same-forward scene/Path features.
# No GPU burn and no scorer training are performed by this stage.
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
LEAD_PYTHON="${B3B_JOINT_PYTHON:-/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead/bin/python}"
CKPT_DIR="${B3B_JOINT_CKPT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB2_corridor}"
DENSE_ROOT="${B3B_JOINT_DENSE_ROOT:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_v2_dense_data}"
SPLIT_DIR="${B3B_JOINT_SPLIT_DIR:-${DENSE_ROOT}/route_split}"
NEAREST_MANIFEST="${B3B_JOINT_NEAREST_MANIFEST:-${REPO_ROOT}/data/p4/manifest.jsonl}"
OUTPUT_ROOT="${B3B_JOINT_OUTPUT_ROOT:-${REPO_ROOT}/outputs/local_training/p5_stepB3b_joint_scorer/scene_cache}"
CHUNK_SIZE="${B3B_JOINT_CHUNK_SIZE:-12000}"
BATCH_SIZE="${B3B_JOINT_BATCH_SIZE:-32}"
NUM_WORKERS="${B3B_JOINT_NUM_WORKERS:-0}"
START="${B3B_JOINT_START:-0}"
LIMIT="${B3B_JOINT_LIMIT:-0}"
SPLIT="${1:-both}"

if [[ "$#" -gt 1 ]] || [[ "${SPLIT}" != "train" && "${SPLIT}" != "heldout" && "${SPLIT}" != "both" ]]; then
    echo "usage: bash $0 [train|heldout|both]" >&2
    exit 2
fi
if [[ ! -x "${LEAD_PYTHON}" || ! -f "${CKPT_DIR}/model_0019.pth" || ! -f "${NEAREST_MANIFEST}" ]]; then
    echo "missing Python, B2 corridor checkpoint, or sparse VLM manifest" >&2
    exit 2
fi
for number in "${CHUNK_SIZE}" "${BATCH_SIZE}" "${START}" "${LIMIT}" "${NUM_WORKERS}"; do
    if [[ ! "${number}" =~ ^[0-9]+$ ]]; then
        echo "chunk/batch/start/limit/workers must be nonnegative integers" >&2
        exit 2
    fi
done
if (( CHUNK_SIZE < 1 || BATCH_SIZE < 1 )); then
    echo "chunk and batch sizes must be positive" >&2
    exit 2
fi

GPU_LIST="${B3B_JOINT_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "${GPU_LIST}" ]]; then
    GPU_COUNT=$("${LEAD_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')
    if (( GPU_COUNT < 1 )); then echo "no CUDA GPU detected" >&2; exit 2; fi
    GPU_LIST=$(seq -s, 0 "$((GPU_COUNT - 1))")
fi
IFS=',' read -r -a DEVICES <<< "${GPU_LIST}"
if (( ${#DEVICES[@]} < 1 )); then echo "empty GPU list" >&2; exit 2; fi
for gpu in "${DEVICES[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
        echo "GPU ids must be numeric: ${gpu}" >&2
        exit 2
    fi
done

cd "${REPO_ROOT}"
extract_split() {
    local split="$1"
    local manifest="${SPLIT_DIR}/${split}.jsonl"
    local output_dir="${OUTPUT_ROOT}/${split}"
    local logs="${OUTPUT_ROOT}/logs/${split}"
    if [[ ! -f "${manifest}" ]]; then echo "missing ${manifest}" >&2; return 2; fi
    mkdir -p "${output_dir}" "${logs}"
    # VLMIntentDataset only keeps dense frames whose route has a sparse VLM
    # cache. Count exactly that subset before slicing valid_indices.
    local total
    total=$("${LEAD_PYTHON}" - "${manifest}" "${NEAREST_MANIFEST}" <<'PY'
import json, sys
from pathlib import Path
manifest, sparse = map(Path, sys.argv[1:])
with sparse.open() as handle:
    routes = {(e['scenario'], e['route']) for e in
              (json.loads(line) for line in handle if line.strip())}
with manifest.open() as handle:
    count = sum((e['scenario'], e['route']) in routes for e in
                (json.loads(line) for line in handle if line.strip()))
print(count)
PY
)
    local end_limit="${total}"
    if (( START >= total )); then echo "start ${START} >= ${split} frames ${total}" >&2; return 2; fi
    if (( LIMIT > 0 && START + LIMIT < total )); then end_limit=$((START + LIMIT)); fi
    local starts=()
    local skipped=0
    for ((start=START; start<end_limit; start+=CHUNK_SIZE)); do
        local end=$((start + CHUNK_SIZE))
        if (( end > end_limit )); then end=${end_limit}; fi
        local output
        output=$(printf '%s/joint_scene_%06d_%06d.npz' "${output_dir}" "${start}" "${end}")
        if [[ -f "${output}" ]]; then
            skipped=$((skipped + 1))
        else
            starts+=("${start}")
        fi
    done
    echo "${split}: eligible=${total}, interval=[${START},${end_limit}), skipped=${skipped}, pending=${#starts[@]}"
    for ((offset=0; offset<${#starts[@]}; offset+=${#DEVICES[@]})); do
        local pids=()
        local labels=()
        for ((slot=0; slot<${#DEVICES[@]} && offset+slot<${#starts[@]}; slot++)); do
            local start=${starts[$((offset+slot))]}
            local end=$((start + CHUNK_SIZE))
            if (( end > end_limit )); then end=${end_limit}; fi
            local log
            log=$(printf '%s/joint_scene_%06d_%06d.log' "${logs}" "${start}" "${end}")
            echo "extract ${split} [${start},${end}) on GPU ${DEVICES[$slot]} -> ${log}"
            CUDA_VISIBLE_DEVICES="${DEVICES[$slot]}" "${LEAD_PYTHON}" \
                scripts/p4/extract_b3b_joint_scene.py \
                --ckpt-dir "${CKPT_DIR}" --manifest "${manifest}" \
                --nearest-vlm-manifest "${NEAREST_MANIFEST}" \
                --output-dir "${output_dir}" --start "${start}" --end "${end}" \
                --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
                >"${log}" 2>&1 &
            pids+=("$!")
            labels+=("${log}")
        done
        local failed=0
        for ((slot=0; slot<${#pids[@]}; slot++)); do
            if ! wait "${pids[$slot]}"; then
                echo "failed: ${labels[$slot]}" >&2
                tail -n 25 "${labels[$slot]}" >&2
                failed=1
            fi
        done
        if (( failed )); then return 1; fi
    done
    "${LEAD_PYTHON}" scripts/p4/verify_b3b_joint_scene_cache.py \
        --cache-dir "${output_dir}" --ckpt-dir "${CKPT_DIR}" \
        --manifest "${manifest}" --nearest-vlm-manifest "${NEAREST_MANIFEST}" \
        --start "${START}" --end "${end_limit}" --chunk-size "${CHUNK_SIZE}"
}

if [[ "${SPLIT}" == "train" || "${SPLIT}" == "both" ]]; then extract_split train; fi
if [[ "${SPLIT}" == "heldout" || "${SPLIT}" == "both" ]]; then extract_split heldout; fi
echo "joint scene cache ready: ${OUTPUT_ROOT}"
