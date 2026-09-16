#!/usr/bin/env bash
# B3a-1b: evaluate current-speed-relative vocabularies with manual profile union.
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
LEAD_ENV="/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead"
CKPT_DIR="${B3A_CKPT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB2_corridor}"
CACHE_DIR="${B3A_CACHE_DIR:-${CKPT_DIR}/b2d_safety_features/heldout_future_actors}"
PROFILE_DIR="${B3A_PROFILE_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_velocity_oracle}"
ABSOLUTE_VOCAB_DIR="${B3A_VELOCITY_VOCAB_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_velocity_vocab}"
RELATIVE_VOCAB_DIR="${B3A_RELATIVE_VOCAB_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_relative_velocity_vocab}"
OUTPUT_DIR="${B3A_RELATIVE_ORACLE_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_relative_velocity_vocab_oracle}"

if [[ ! -x "${LEAD_ENV}/bin/python" ]]; then
    echo "lead environment not found: ${LEAD_ENV}" >&2
    exit 1
fi
if [[ ! -f "${PROFILE_DIR}/velocity_profile_oracle.frames.npz" ]]; then
    echo "B3a-0b frame records not found: ${PROFILE_DIR}" >&2
    exit 1
fi
if [[ ! -f "${RELATIVE_VOCAB_DIR}/relative_velocity_vocab_k64.npy" ]]; then
    echo "relative vocabulary not found: ${RELATIVE_VOCAB_DIR}" >&2
    echo "Run scripts/p4/build_p5_stepB3a_relative_velocity_vocab.sh first." >&2
    exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"
"${LEAD_ENV}/bin/python" scripts/p4/eval_b3a_velocity_vocab_oracle.py \
    --base-frames "${PROFILE_DIR}/velocity_profile_oracle.frames.npz" \
    --base-report "${PROFILE_DIR}/velocity_profile_oracle.json" \
    --vocab-dir "${RELATIVE_VOCAB_DIR}" \
    --expert-profiles "${ABSOLUTE_VOCAB_DIR}/heldout_velocity_profiles.npz" \
    --future-cache-dir "${CACHE_DIR}" \
    --representation current-relative \
    --clusters 16,32,64 \
    --limit 5000 \
    --out "${OUTPUT_DIR}/relative_velocity_vocab_oracle.json" \
    "$@"
