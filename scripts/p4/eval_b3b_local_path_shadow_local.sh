#!/usr/bin/env bash
# Closed-loop local-Path shadow: execute baseline, log counterfactuals only.
# Usage: B2D_ROUTE_IDS="2084" bash this_script <corridor_ckpt> "0" [tag]
set -euo pipefail

CHECKPOINT_DIR="${1:?provide the B2 corridor checkpoint directory}"
GPU_LIST="${2:-0}"
OUTPUT_TAG="${3:-b3b_local_path_shadow}"

export LEAD_TRAINING_CONFIG="${LEAD_TRAINING_CONFIG:-} route_selection_mode=confidence route_local_path_shadow=true route_predicted_actor_velocity_gate=false route_predicted_actor_velocity_shadow=false route_velocity_scorer_gate=false route_speed_safety_gate=false route_future_safety_gate=false"

bash scripts/eval_bench2drive_local.sh "${CHECKPOINT_DIR}" "${GPU_LIST}" "${OUTPUT_TAG}"
