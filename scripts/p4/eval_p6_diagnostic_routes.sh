#!/usr/bin/env bash
# Trace four representative P6 single-speed routes without changing policy.
# Usage: bash scripts/p4/eval_p6_diagnostic_routes.sh <checkpoint.pth> "0 1 2 3" <unique-tag>
set -euo pipefail

CKPT="${1:?provide P6 single-speed checkpoint}"
GPUS="${2:-0}"
TAG="${3:?provide a new output tag}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export B2D_ROUTE_IDS="${P6_B2D_DIAGNOSTIC_ROUTES:-3904 3737 2881 24340}"
export P6_B2D_TRACE=1
export P6_B2D_TRACE_EVERY="${P6_B2D_TRACE_EVERY:-1}"
export P6_B2D_TRACE_MAP_EVERY="${P6_B2D_TRACE_MAP_EVERY:-5}"
export ROUTE_TIMEOUT="${ROUTE_TIMEOUT:-3600}"
echo "Diagnostic routes: $B2D_ROUTE_IDS"
result=0
bash scripts/p4/eval_p6_joint_b2d.sh "$CKPT" "$GPUS" "$TAG" || result=$?

LEAD_PYTHON="${P6_B2D_LEAD_PYTHON:-python}"
"$LEAD_PYTHON" scripts/p4/summarize_p6_candidate_trace.py "outputs/local_evaluation_$TAG" || true
exit "$result"
