#!/usr/bin/env bash
# Matched closed-loop control arm for eval_b3a_predicted_actor_gate_local.sh.
set -euo pipefail

CHECKPOINT_DIR="${1:?provide the corridor checkpoint directory}"
GPU_LIST="${2:-0}"
OUTPUT_TAG="${3:-b3a_actor_baseline}"

export LEAD_TRAINING_CONFIG="${LEAD_TRAINING_CONFIG:-} route_selection_mode=confidence route_predicted_actor_velocity_gate=false route_velocity_scorer_gate=false route_speed_safety_gate=false route_future_safety_gate=false"

bash scripts/eval_bench2drive_local.sh "${CHECKPOINT_DIR}" "${GPU_LIST}" "${OUTPUT_TAG}"
