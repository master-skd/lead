#!/usr/bin/env bash
# Shard the B3b GT oracle across visible GPUs, then merge frame-level records.
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
LEAD_PYTHON="/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead/bin/python"
GPU_LIST="${B3B_GPU_LIST:-0,1,2,3,4,5,6,7}"
START="${B3B_START:-0}"
LIMIT="${B3B_LIMIT:-5000}"
OUT="${B3B_OUTPUT:-${REPO_ROOT}/outputs/local_training/p5_stepB3b_joint_oracle/joint_k16_local_8gpu.json}"
LOCAL_OFFSETS="${B3B_LOCAL_OFFSETS:--1.5,-0.75,0.75,1.5}"
CPU_THREADS="${B3B_CPU_THREADS:-4}"

if [[ ! "${START}" =~ ^[0-9]+$ || ! "${LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "B3B_START must be nonnegative and B3B_LIMIT must be positive" >&2
    exit 2
fi
if [[ ! "${CPU_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "B3B_CPU_THREADS must be positive" >&2
    exit 2
fi
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if (( ${#GPUS[@]} == 0 )); then
    echo "B3B_GPU_LIST is empty" >&2
    exit 2
fi
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ || -v "SEEN[${gpu}]" ]]; then
        echo "B3B_GPU_LIST must contain distinct numeric GPU IDs" >&2
        exit 2
    fi
    SEEN["${gpu}"]=1
done
if [[ ! -x "${LEAD_PYTHON}" ]]; then
    echo "missing lead Python: ${LEAD_PYTHON}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
OUT_DIR="$(dirname "${OUT}")"
mkdir -p "${OUT_DIR}/shards"
OUT_STEM="$(basename "${OUT}" .json)"
PIDS=()
SHARDS=()
LOGS=()
cleanup() {
    for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
}
trap cleanup INT TERM
for i in "${!GPUS[@]}"; do
    shard_start=$(( START + LIMIT * i / ${#GPUS[@]} ))
    shard_end=$(( START + LIMIT * (i + 1) / ${#GPUS[@]} ))
    if (( shard_end == shard_start )); then continue; fi
    shard="${OUT_DIR}/shards/${OUT_STEM}_${shard_start}_${shard_end}.json"
    log="${OUT_DIR}/shards/${OUT_STEM}_${shard_start}_${shard_end}.log"
    echo "GPU ${GPUS[i]}: frames [${shard_start},${shard_end}) -> ${shard}; log=${log}"
    CUDA_VISIBLE_DEVICES="${GPUS[i]}" B3B_OUTPUT="${shard}" \
        OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" \
        OPENBLAS_NUM_THREADS="${CPU_THREADS}" \
        bash scripts/p4/eval_b3b_joint_oracle.sh \
        --start "${shard_start}" --limit "$((shard_end - shard_start))" \
        "--local-offsets=${LOCAL_OFFSETS}" "$@" >"${log}" 2>&1 &
    PIDS+=("$!")
    SHARDS+=("${shard}")
    LOGS+=("${log}")
done
failed=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[i]}"; then
        echo "shard failed: ${SHARDS[i]}; inspect ${LOGS[i]}" >&2
        failed=1
    fi
done
trap - INT TERM
if (( failed )); then exit 1; fi
"${LEAD_PYTHON}" scripts/p4/merge_b3b_joint_oracle.py --out "${OUT}" "${SHARDS[@]}"
"${LEAD_PYTHON}" scripts/p4/audit_b3b_local_paths.py \
    --report "${OUT}" --out "${OUT%.json}.feasibility.json"
