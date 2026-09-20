#!/usr/bin/env bash
# Evaluate direct velocity-profile selection with the existing corridor/scorer weights.
# Usage: bash scripts/p4/eval_b3a_profile_select_local.sh <checkpoint_dir> "0 1 2 3 4 5 6 7" [tag]
set -euo pipefail

CHECKPOINT_DIR="${1:?provide the corridor checkpoint directory}"
GPU_LIST="${2:-0}"
OUTPUT_TAG="${3:-b3a_profile_select}"

# TrainingConfig applies this override after loading the checkpoint's config.json.
# Keep the B2c/B2d speed gates disabled so this is a clean velocity-only comparison.
export LEAD_TRAINING_CONFIG="${LEAD_TRAINING_CONFIG:-} route_selection_mode=confidence route_velocity_scorer_gate=true route_velocity_selection_mode=profile_select route_speed_safety_gate=false route_future_safety_gate=false"

bash scripts/eval_bench2drive_local.sh "$CHECKPOINT_DIR" "$GPU_LIST" "$OUTPUT_TAG"
