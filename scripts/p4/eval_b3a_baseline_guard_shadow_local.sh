#!/usr/bin/env bash
# Run the conservative velocity scorer in shadow: execute the unchanged corridor
# target while logging the speed it would have selected.
# Usage: B2D_ROUTE_IDS="2084 2091" bash this_script <ckpt_dir> "0" [tag]
set -euo pipefail

CHECKPOINT_DIR="${1:?provide the corridor checkpoint directory}"
GPU_LIST="${2:-0}"
OUTPUT_TAG="${3:-b3a_baseline_guard_shadow}"
SCORER="${B3A_GUARD_SCORER:-outputs/local_training/p5_stepB3a_velocity_scorer/velocity_scorer_best.pth}"
VOCABULARY="${B3A_GUARD_VOCABULARY:-outputs/local_training/p5_stepB3a_relative_velocity_vocab/relative_velocity_vocab_k64.npy}"

if [[ ! -f "${SCORER}" || ! -f "${VOCABULARY}" ]]; then
    echo "missing B3a scorer or vocabulary: ${SCORER} ${VOCABULARY}" >&2
    exit 1
fi

export LEAD_TRAINING_CONFIG="${LEAD_TRAINING_CONFIG:-} route_selection_mode=confidence route_velocity_scorer_gate=true route_velocity_selection_mode=baseline_guard route_velocity_baseline_guard_shadow=true route_velocity_scorer_head=${SCORER} route_velocity_vocabulary=${VOCABULARY} route_velocity_baseline_guard_unsafe_threshold=0.7 route_velocity_baseline_guard_safe_threshold=0.5 route_velocity_baseline_guard_max_slowdown_mps=1.5 route_predicted_actor_velocity_gate=false route_speed_safety_gate=false route_future_safety_gate=false"

echo "baseline_guard shadow only: the executed target speed stays unchanged"
bash scripts/eval_bench2drive_local.sh "${CHECKPOINT_DIR}" "${GPU_LIST}" "${OUTPUT_TAG}"
