#!/usr/bin/env bash
# CPU-only, resumable GT label construction for frozen B2 same-pass scene shards.
set -euo pipefail

ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
PYTHON="${B3B_PYTHON:-/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead/bin/python}"
SCENE_ROOT="${B3B_SCENE_ROOT:-${ROOT}/outputs/local_training/p5_stepB3b_joint_scorer/scene_cache}"
ACTOR_ROOT="${B3B_ACTOR_ROOT:-${ROOT}/outputs/local_training/p5_stepB3a_v2_dense_data/future_actor_cache}"
LABEL_ROOT="${B3B_LABEL_ROOT:-${ROOT}/outputs/local_training/p5_stepB3b_joint_scorer/labels}"
HDMAP_ROOT="${B3B_HDMAP_ROOT:-${ROOT}/data/carla_leaderboard2/data}"
WORKERS="${B3B_LABEL_WORKERS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
SPLIT="${1:-both}"

if [[ "${SPLIT}" != "train" && "${SPLIT}" != "heldout" && "${SPLIT}" != "both" ]]; then
    echo "usage: bash $0 [train|heldout|both]" >&2
    exit 2
fi
cd "${ROOT}"
run_split() {
    local split="$1"
    "${PYTHON}" scripts/p4/build_b3b_joint_labels.py \
        --all --scene-dir "${SCENE_ROOT}/${split}" \
        --actor-dir "${ACTOR_ROOT}/${split}_future_actors" \
        --actor-index "${LABEL_ROOT}/${split}_actor_index.sqlite" \
        --hdmap-root "${HDMAP_ROOT}" \
        --output-dir "${LABEL_ROOT}/${split}" --workers "${WORKERS}"
}
if [[ "${SPLIT}" == "both" ]]; then
    run_split train
    run_split heldout
else
    run_split "${SPLIT}"
fi
echo "B3b joint GT labels ready: ${LABEL_ROOT}"
