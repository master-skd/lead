#!/bin/bash
# Single-machine multi-GPU Bench2Drive eval driver (no SLURM).
# Splits the 220 routes across GPUs; each GPU runs its own CARLA (Epic quality) + agent.
#
# Usage:
#   bash scripts/eval_bench2drive_local.sh <ckpt_dir> "<gpu list>" [out_tag]
#   e.g. bash scripts/eval_bench2drive_local.sh outputs/checkpoints/my_p2 "0 1 2 3 4 5 6 7" my_p2
#
# Prereqs: env set (LEAD_PROJECT_ROOT, CARLA_ROOT), HF offline, resnet34 weights cached.
set -u
CKPT="${1:?checkpoint dir, e.g. outputs/checkpoints/my_p2}"
GPUS="${2:-0}"
TAG="${3:-run}"
# Per-route wallclock cap. Overridable because 1500 s is NOT enough on a loaded machine:
# measured ratio was 0.133x real time (7.5x slower), so 1500 s buys only ~200 s of game
# time and any longer route gets SIGTERMed mid-drive. That looks like a silent model
# failure in the log -- `timeout` sends SIGTERM, Python has no handler, so there is no
# traceback, just a log that stops. The route then has no checkpoint_endpoint.json and
# scores ZERO, since merge_route_json.py divides by a hardcoded 220 rather than by the
# number of routes actually completed (~0.43 DS per lost route).
ROUTE_TIMEOUT="${ROUTE_TIMEOUT:-1500}"
ROUTES_DIR="data/benchmark_routes/bench2drive"
LOGDIR="/tmp/b2d_${TAG}"
# Per-model output root so two models can eval the SAME 220 route ids simultaneously
# without colliding on outputs/local_evaluation/<id>/ (which is keyed by route id only).
OUTROOT="outputs/local_evaluation_${TAG}"
mkdir -p "$LOGDIR" "$OUTROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mapfile -t ROUTES < <(ls "$ROUTES_DIR"/*.xml | sort)
GPU_ARR=($GPUS); NG=${#GPU_ARR[@]}
echo "routes=${#ROUTES[@]}  gpus=$NG  ckpt=$CKPT"

wait_port()  { for _ in $(seq 1 60); do ss -ltn | grep -q ":$1 " && return 0; sleep 3; done; return 1; }
wait_free()  { for _ in $(seq 1 30); do ss -ltn | grep -q ":$1 " || return 0; sleep 1; done; return 1; }

start_carla() {  # $1 gpu  $2 port  $3 strm -> echoes pid
  local gpu=$1 port=$2 strm=$3
  CUDA_VISIBLE_DEVICES=$gpu "$CARLA_ROOT/CarlaUE4.sh" --world-port=$port \
      --carla-streaming-port=$strm -nosound -graphicsadapter=$gpu -RenderOffScreen \
      >"$LOGDIR/carla_gpu${gpu}.log" 2>&1 &
  echo $!
}

kill_carla() {  # $1 pid  $2 port  $3 gpu -- kill and wait for the RPC port to free
  local pid=$1 port=$2 gpu=$3
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null
  pkill -9 -f "graphicsadapter=$gpu" 2>/dev/null   # sweep any orphaned UE child on this adapter
  wait_free "$port"
}

worker() {
  local gpu=$1; shift
  # CARLA needs 3 consecutive ports (world, +1 secondary/streaming); stride 10 avoids overlap.
  local port=$((2000 + gpu*10)) strm=$((2000 + gpu*10 + 1))
  local ok=0 fail=0 i=0
  # ROTATE the traffic-manager port per route. CARLA's TM RPC socket lacks SO_REUSEADDR, so
  # reusing a TM port across per-route restarts hits TCP TIME_WAIT (~60s) -> bind error.
  # Each GPU owns a 400-wide band (base 31000); we rotate a 190-wide window inside it, and
  # start at a time-seeded offset so a *fresh run* also never lands on the previous run's
  # still-cooling ports. seed varies by wall-clock + gpu.
  local tmbase=$((31000 + gpu*400))
  local off=$(( ( $(date +%s) + gpu*97 ) % 190 ))
  # Fresh CARLA PER ROUTE: a long-lived CARLA leaks/segfaults after ~dozens of routes and
  # then poisons every remaining route on that GPU. Restarting per route isolates crashes.
  for xml in "$@"; do
    local tm=$(( tmbase + (off + i) % 190 )); i=$((i+1))
    local id; id=$(basename "$xml" .xml)
    [ -f "$OUTROOT/$id/checkpoint_endpoint.json" ] && { echo "[gpu$gpu] $id done, skip"; continue; }
    # make sure the ports are clear before we boot (belt-and-suspenders vs a prior straggler)
    pkill -9 -f "graphicsadapter=$gpu" 2>/dev/null; wait_free "$port"
    local carla_pid; carla_pid=$(start_carla $gpu $port $strm)
    if ! wait_port $port; then
      echo "[gpu$gpu] $id: CARLA failed to bind $port, retry once"
      kill_carla "$carla_pid" "$port" "$gpu"
      carla_pid=$(start_carla $gpu $port $strm)
      if ! wait_port $port; then
        echo "[gpu$gpu] $id CARLA-DOWN (see $LOGDIR/carla_gpu${gpu}.log)"; fail=$((fail+1))
        kill_carla "$carla_pid" "$port" "$gpu"; continue
      fi
    fi
    echo "[gpu$gpu] route $id (carla up on $port) ..."
    if CUDA_VISIBLE_DEVICES=$gpu timeout "$ROUTE_TIMEOUT" python -m lead \
        --checkpoint "$CKPT" --routes "$xml" --bench2drive \
        --output-dir "$OUTROOT/$id" \
        --port $port --traffic-manager-port $tm \
        >"$LOGDIR/route_${id}.log" 2>&1; then ok=$((ok+1))
    else
      # Distinguish "ran out of time" from "crashed": `timeout` exits 124 on SIGTERM, and a
      # timed-out run leaves NO traceback (Python installs no SIGTERM handler), so the log
      # just stops mid-line and reads exactly like a silent model failure. Say which it was.
      rc=$?
      if [ "$rc" -eq 124 ]; then
        echo "[gpu$gpu] $id TIMEOUT after ${ROUTE_TIMEOUT}s -- route unfinished, will score 0." \
             "Raise ROUTE_TIMEOUT and re-run with the SAME tag to fill it in."
      else
        echo "[gpu$gpu] $id FAILED rc=$rc (see $LOGDIR/route_${id}.log)"
      fi
      fail=$((fail+1))
    fi
    kill_carla "$carla_pid" "$port" "$gpu"
  done
  echo "[gpu$gpu] finished: ok=$ok fail=$fail"
}

for i in "${!GPU_ARR[@]}"; do
  gpu=${GPU_ARR[$i]}; sub=()
  for j in "${!ROUTES[@]}"; do (( j % NG == i )) && sub+=("${ROUTES[$j]}"); done
  worker "$gpu" "${sub[@]}" &
done
wait
echo "=== all workers done. logs in $LOGDIR ==="
echo "collect results + merge:"
echo "  mkdir -p outputs/b2d_${TAG} && for f in $OUTROOT/*/checkpoint_endpoint.json; do cp \$f outputs/b2d_${TAG}/\$(basename \$(dirname \$f)).json; done"
echo "  python slurm/evaluation/merge_route_json.py -f outputs/b2d_${TAG}"
