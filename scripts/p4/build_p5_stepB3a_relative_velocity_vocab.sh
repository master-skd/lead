#!/usr/bin/env bash
# Build current-speed-relative expert velocity vocabularies from cached profiles.
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
LEAD_ENV="/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead"
SOURCE_DIR="${B3A_VELOCITY_VOCAB_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_velocity_vocab}"
OUTPUT_DIR="${B3A_RELATIVE_VOCAB_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_relative_velocity_vocab}"

cd "${REPO_ROOT}"
export OMP_NUM_THREADS="${B3A_KMEANS_THREADS:-16}"
"${LEAD_ENV}/bin/python" scripts/p4/build_b3a_relative_velocity_vocab.py \
    --source-dir "${SOURCE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
