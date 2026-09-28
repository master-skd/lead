#!/usr/bin/env bash
# GT-privileged Path x (raw + K Velocity) candidate-coverage audit.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/p4/eval_b3b_joint_oracle.sh --limit 5000
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
LEAD_ENV="/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead"
DENSE_ROOT="${B3B_DENSE_ROOT:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_v2_dense_data}"
CKPT_DIR="${B3B_CKPT_DIR:-${REPO_ROOT}/outputs/local_training/p5_stepB2_corridor}"
OUT="${B3B_OUTPUT:-${REPO_ROOT}/outputs/local_training/p5_stepB3b_joint_oracle/joint_oracle.json}"
VOCAB_K="${B3B_VOCAB_K:-16}"
if [[ -n "${B3B_VOCAB_PATH:-}" ]]; then
    VOCAB_PATH="${B3B_VOCAB_PATH}"
else
    case "${VOCAB_K}" in
        16|32|64) ;;
        *) echo "B3B_VOCAB_K must be 16, 32 or 64" >&2; exit 2 ;;
    esac
    VOCAB_PATH="${DENSE_ROOT}/relative_velocity_vocab/relative_velocity_vocab_k${VOCAB_K}.npy"
fi

for required in \
    "${CKPT_DIR}/model_0019.pth" \
    "${DENSE_ROOT}/route_split/heldout.jsonl" \
    "${VOCAB_PATH}"; do
    if [[ ! -f "${required}" ]]; then
        echo "missing B3b input: ${required}" >&2
        exit 1
    fi
done
if [[ ! -x "${LEAD_ENV}/bin/python" ]]; then
    echo "missing lead environment: ${LEAD_ENV}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
"${LEAD_ENV}/bin/python" scripts/p4/eval_b3b_joint_oracle.py \
    --ckpt-dir "${CKPT_DIR}" \
    --manifest "${DENSE_ROOT}/route_split/heldout.jsonl" \
    --nearest-vlm-manifest "${B3B_NEAREST_VLM_MANIFEST:-${REPO_ROOT}/data/p4/manifest.jsonl}" \
    --future-cache-dir "${DENSE_ROOT}/future_actor_cache/heldout_future_actors" \
    --velocity-vocab "${VOCAB_PATH}" \
    --out "${OUT}" \
    "$@"
