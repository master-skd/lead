#!/usr/bin/env bash
# Build route-disjoint dense (stride=1) manifests and velocity vocabularies.
# Qwen-VL is not rerun: feature extraction reuses the nearest sparse cached intent
# on the same route, while RGB/LiDAR/actors/speed labels remain from the exact frame.
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
LEAD_ENV="/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead"
DENSE_ROOT="${B3A_V2_DENSE_ROOT:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_v2_dense_data}"
MANIFEST="${DENSE_ROOT}/manifest_stride1.jsonl"
SPLIT_DIR="${DENSE_ROOT}/route_split"
VOCAB_DIR="${DENSE_ROOT}/velocity_vocab"
RELATIVE_DIR="${DENSE_ROOT}/relative_velocity_vocab"

mkdir -p "${DENSE_ROOT}"
cd "${REPO_ROOT}"

if [[ ! -f "${MANIFEST}" ]]; then
    CUDA_VISIBLE_DEVICES="" "${LEAD_ENV}/bin/python" scripts/p4/dump_front_images.py \
        --manifest "${MANIFEST}" --stride 1 --skip-meta
fi

if [[ ! -f "${SPLIT_DIR}/train.jsonl" || ! -f "${SPLIT_DIR}/heldout.jsonl" ]]; then
    "${LEAD_ENV}/bin/python" scripts/p4/make_b2d_route_split.py \
        --manifest "${MANIFEST}" --output-dir "${SPLIT_DIR}" \
        --heldout-fraction 0.1 --seed 20260901
fi

B3A_SPLIT_DIR="${SPLIT_DIR}" \
B3A_VELOCITY_VOCAB_DIR="${VOCAB_DIR}" \
B3A_SKIP_GPU_BURN=1 \
    bash scripts/p4/build_p5_stepB3a_velocity_vocab.sh "$@"

B3A_VELOCITY_VOCAB_DIR="${VOCAB_DIR}" \
B3A_RELATIVE_VOCAB_DIR="${RELATIVE_DIR}" \
    bash scripts/p4/build_p5_stepB3a_relative_velocity_vocab.sh

echo "B3a-v2 dense data ready: ${DENSE_ROOT}"
