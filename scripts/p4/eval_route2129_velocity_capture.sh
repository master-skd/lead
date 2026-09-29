#!/usr/bin/env bash
# Read-only route 2129 trace. Run shadow first, then active only if the
# counterfactual speed command merits an actuation comparison.
# Usage: bash scripts/p4/eval_route2129_velocity_capture.sh shadow|active <ckpt_dir> [gpu] [tag]
set -euo pipefail

MODE="${1:?choose shadow or active}"
CHECKPOINT_DIR="${2:?provide the corridor checkpoint directory}"
GPU="${3:-0}"
TAG="${4:-route2129_velocity_${MODE}}"
SCORER="${B3A_GUARD_SCORER:-outputs/local_training/p5_stepB3a_velocity_scorer/velocity_scorer_best.pth}"
VOCABULARY="${B3A_GUARD_VOCABULARY:-outputs/local_training/p5_stepB3a_relative_velocity_vocab/relative_velocity_vocab_k64.npy}"

case "$MODE" in
  shadow) SHADOW=true ;;
  active) SHADOW=false ;;
  *) echo "mode must be shadow or active" >&2; exit 2 ;;
esac
for path in "$SCORER" "$VOCABULARY"; do
  if [[ ! -f "$path" ]]; then
    echo "missing scorer/vocabulary: $path" >&2
    exit 1
  fi
done

export B2D_ROUTE_IDS=2129
export LEAD_ROUTE2129_CAPTURE=1
export LEAD_ROUTE2129_CAPTURE_START="${LEAD_ROUTE2129_CAPTURE_START:-180}"
export LEAD_ROUTE2129_CAPTURE_END="${LEAD_ROUTE2129_CAPTURE_END:-360}"
export LEAD_TRAINING_CONFIG="${LEAD_TRAINING_CONFIG:-} route_selection_mode=confidence route_velocity_scorer_gate=true route_velocity_selection_mode=baseline_guard route_velocity_baseline_guard_shadow=${SHADOW} route_velocity_scorer_head=${SCORER} route_velocity_vocabulary=${VOCABULARY} route_velocity_baseline_guard_unsafe_threshold=0.7 route_velocity_baseline_guard_safe_threshold=0.5 route_velocity_baseline_guard_max_slowdown_mps=1.5 route_predicted_actor_velocity_gate=false route_speed_safety_gate=false route_future_safety_gate=false"

echo "route 2129 mode=$MODE capture_steps=$LEAD_ROUTE2129_CAPTURE_START:$LEAD_ROUTE2129_CAPTURE_END"
bash scripts/eval_bench2drive_local.sh "$CHECKPOINT_DIR" "$GPU" "$TAG"
echo "capture: outputs/local_evaluation_${TAG}/2129/route2129_capture.jsonl"
echo "decisions: outputs/local_evaluation_${TAG}/2129/velocity_diagnostics.jsonl"
