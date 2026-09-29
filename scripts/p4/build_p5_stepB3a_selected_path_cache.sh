#!/usr/bin/env bash
# Re-forward the frozen corridor model only to recover full selected Paths.
# Existing velocity/actor/scene-label shards are never rewritten.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LEAD_ENV="${LEAD_ENV:-/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead}"
CKPT_DIR="${B3A_CKPT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB2_corridor}"
DENSE_ROOT="${B3A_V2_DENSE_ROOT:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_v2_dense_data}"
FEATURE_ROOT="${B3A_V2_FEATURE_ROOT:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_v2_scene_scorer/features}"
PATH_ROOT="${B3A_SELECTED_PATH_ROOT:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_v2_scene_scorer/selected_paths}"
NEAREST_VLM_MANIFEST="${B3A_NEAREST_VLM_MANIFEST:-${REPO_ROOT}/data/p4/manifest.jsonl}"
BATCH_SIZE="${B3A_EXTRACT_BATCH_SIZE:-64}"
NUM_WORKERS="${B3A_EXTRACT_NUM_WORKERS:-4}"
read -r -a SPLITS <<< "${B3A_PATH_SPLITS:-heldout train}"

if [[ ! -x "${LEAD_ENV}/bin/python" || ! -f "${CKPT_DIR}/model_0019.pth" ]]; then
    echo "missing lead Python or corridor checkpoint" >&2
    exit 1
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
else
    GPU_COUNT="$("${LEAD_ENV}/bin/python" -c 'import torch; print(torch.cuda.device_count())')"
    if (( GPU_COUNT < 1 )); then
        echo "no CUDA device available" >&2
        exit 1
    fi
    DEVICES=()
    for ((gpu=0; gpu<GPU_COUNT; gpu++)); do DEVICES+=("${gpu}"); done
fi
if (( ${#DEVICES[@]} < 1 )); then
    echo "CUDA_VISIBLE_DEVICES did not contain a GPU" >&2
    exit 1
fi

cd "${REPO_ROOT}"
for split in "${SPLITS[@]}"; do
    if [[ "${split}" != "train" && "${split}" != "heldout" ]]; then
        echo "invalid split: ${split}" >&2
        exit 2
    fi
    FEATURE_DIR="${FEATURE_ROOT}/${split}"
    PATH_DIR="${PATH_ROOT}/${split}"
    MANIFEST="${DENSE_ROOT}/route_split/${split}.jsonl"
    if [[ ! -f "${MANIFEST}" ]]; then
        echo "missing manifest: ${MANIFEST}" >&2
        exit 1
    fi
    mkdir -p "${PATH_DIR}"
    shopt -s nullglob
    FEATURE_FILES=("${FEATURE_DIR}"/velocity_features_*.npz)
    shopt -u nullglob
    if (( ${#FEATURE_FILES[@]} < 1 )); then
        echo "no existing velocity feature shards in ${FEATURE_DIR}" >&2
        exit 1
    fi
    PENDING=()
    skipped=0
    for feature in "${FEATURE_FILES[@]}"; do
        basename="$(basename "${feature}")"
        if [[ ! "${basename}" =~ ^velocity_features_([0-9]{6})_([0-9]{6})\.npz$ ]]; then
            echo "unexpected feature shard name: ${feature}" >&2
            exit 1
        fi
        start="${BASH_REMATCH[1]}"
        end="${BASH_REMATCH[2]}"
        output="${PATH_DIR}/selected_paths_${start}_${end}.npz"
        if [[ -f "${output}" ]]; then
            skipped=$((skipped + 1))
        else
            PENDING+=("${start}:${end}")
        fi
    done
    echo "${split}: ${#FEATURE_FILES[@]} feature shards; ${skipped} Paths cached; ${#PENDING[@]} pending"
    for ((offset=0; offset<${#PENDING[@]}; offset+=${#DEVICES[@]})); do
        pids=()
        for ((slot=0; slot<${#DEVICES[@]} && offset+slot<${#PENDING[@]}; slot++)); do
            IFS=':' read -r start end <<< "${PENDING[$((offset+slot))]}"
            echo "extract ${split} selected Path [${start},${end}) on GPU ${DEVICES[$slot]}"
            CUDA_VISIBLE_DEVICES="${DEVICES[$slot]}" "${LEAD_ENV}/bin/python" \
                scripts/p4/extract_b3a_velocity_features.py \
                --selected-path-only \
                --ckpt-dir "${CKPT_DIR}" \
                --manifest "${MANIFEST}" \
                --nearest-vlm-manifest "${NEAREST_VLM_MANIFEST}" \
                --output-dir "${PATH_DIR}" \
                --start "$((10#${start}))" --end "$((10#${end}))" \
                --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do wait "${pid}"; done
    done
    "${LEAD_ENV}/bin/python" scripts/p4/verify_b3a_selected_paths.py \
        --features "${FEATURE_DIR}" --selected-paths "${PATH_DIR}"
done

echo "selected Path cache complete: ${PATH_ROOT}"
