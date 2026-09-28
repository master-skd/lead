#!/usr/bin/env bash
# Same computation as the active gate, but always execute the baseline speed.
# Usage: B2D_ROUTE_IDS="2084 2091" bash this_script <ckpt_dir> "0 1" [tag]
set -euo pipefail

CHECKPOINT_DIR="${1:?provide the corridor checkpoint directory}"
GPU_LIST="${2:-0}"
OUTPUT_TAG="${3:-b3a_actor_gate_shadow}"
VOCABULARY="${B3A_PRED_ACTOR_VOCABULARY:-outputs/local_training/p5_stepB3a_v2_dense_data/relative_velocity_vocab/relative_velocity_vocab_k64.npy}"

if [[ ! -f "${VOCABULARY}" ]]; then
    echo "missing dense K64 velocity vocabulary: ${VOCABULARY}" >&2
    exit 1
fi

export LEAD_TRAINING_CONFIG="${LEAD_TRAINING_CONFIG:-} route_selection_mode=confidence route_predicted_actor_velocity_gate=true route_predicted_actor_velocity_shadow=true route_predicted_actor_velocity_vocabulary=${VOCABULARY} route_predicted_actor_score_threshold=0.5 route_predicted_actor_nms_iou_threshold=0.5 route_predicted_actor_safety_margin_m=0.2 route_velocity_scorer_gate=false route_speed_safety_gate=false route_future_safety_gate=false"

bash scripts/eval_bench2drive_local.sh "${CHECKPOINT_DIR}" "${GPU_LIST}" "${OUTPUT_TAG}"
