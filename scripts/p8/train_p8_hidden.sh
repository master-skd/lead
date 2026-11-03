#!/usr/bin/env bash
# Frozen Qwen-Drive forward + continuous K-intent planner. Default: fixed 1/10 frames.
set -euo pipefail
cd "$(dirname "$0")/../.."
LEAD_PYTHON=${P8_LEAD_PYTHON:-/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead/bin/python}
QWEN_PYTHON=${P8_QWEN_PYTHON:-/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/qwendrive_official/bin/python}
QWEN_MODEL=${P8_QWEN_MODEL:-/mmu_mllm_hdd_3/liuzihan08/models/Qwen-Drive-1.0-4B}
POSTTRAIN=${P8_POSTTRAIN:-outputs/local_training/posttrain/model_0030.pth}
LOGDIR=${P8_LOGDIR:-outputs/local_training/p8_hidden_tenth}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
BATCH_SIZE=${P8_BATCH_SIZE:-64}
WORKERS=${P8_WORKERS:-8}
EPOCHS=${P8_EPOCHS:-20}
CACHE_DIR=${P8_HIDDEN_CACHE:-}
IFS=',' read -r -a DEVICES <<< "$GPUS"
for value in "$BATCH_SIZE" "$WORKERS" "$EPOCHS" "${P8_QWEN_MICROBATCH:-8}"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "batch/workers/epochs/microbatch must be positive integers" >&2; exit 1; }
done
(( BATCH_SIZE % ${#DEVICES[@]} == 0 )) || { echo 'global batch must be divisible by GPU count' >&2; exit 1; }
[[ -x "$LEAD_PYTHON" \
   && -f "$POSTTRAIN" && -f "$(dirname "$POSTTRAIN")/config.json" \
   && -d data/carla_leaderboard2/data && -d data/carla_leaderboard2/buckets ]] || {
    echo 'Check Python environments, Qwen model, posttrain and CARLA data paths' >&2; exit 1;
}
if [[ -n "$CACHE_DIR" ]]; then
    read -r HIDDEN_DIM FRAME_FRACTION SAMPLE_SEED < <("$LEAD_PYTHON" - "$CACHE_DIR" <<'PYMETA'
import os, sys
from lead.tfv6.hidden_state_cache import HiddenStateCache
cache=HiddenStateCache(sys.argv[1])
sampling=cache.signature['sampling']
for key,value in (('P8_FRAME_FRACTION',sampling['fraction']),('P8_SAMPLE_SEED',sampling['seed'])):
    if key in os.environ and float(os.environ[key])!=value:
        raise ValueError(f'{key} differs from the extracted cache')
print(cache.hidden_dim,sampling['fraction'],sampling['seed'])
cache.close()
PYMETA
    )
    [[ -n "${HIDDEN_DIM:-}" ]] || { echo 'Cache preflight failed' >&2; exit 1; }
else
    [[ -x "$QWEN_PYTHON" && -f "$QWEN_MODEL/config.json" ]] || { echo 'Qwen environment/model missing' >&2; exit 1; }
    HIDDEN_DIM=$("$LEAD_PYTHON" - "$QWEN_MODEL/config.json" <<'PYMETA'
import json, sys
print(json.load(open(sys.argv[1]))['vlm_config']['text_config']['hidden_size'])
PYMETA
    )
    FRAME_FRACTION=${P8_FRAME_FRACTION:-0.1}
    SAMPLE_SEED=${P8_SAMPLE_SEED:-2026}
fi
if [[ "${P8_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    echo "P8 preflight OK: cache=${CACHE_DIR:-online} hidden_dim=$HIDDEN_DIM fraction=$FRAME_FRACTION seed=$SAMPLE_SEED"
    exit 0
fi
mkdir -p "$LOGDIR"
socket_dir=$(mktemp -d "${TMPDIR:-/tmp}/lead_p8_hidden.XXXXXX")
service_pids=()
cleanup() {
    for pid in "${service_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${service_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    for ((i=0; i<${#DEVICES[@]}; i++)); do rm -f "$socket_dir/rank$i.sock"; done
    rmdir "$socket_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LEAD_PROJECT_ROOT="$PWD"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Start synchronously for the first experiment. Optional ordered one-batch overlap.
export P8_HIDDEN_PREFETCH=${P8_HIDDEN_PREFETCH:-0}
if [[ -z "$CACHE_DIR" ]]; then
for ((i=0; i<${#DEVICES[@]}; i++)); do
    CUDA_VISIBLE_DEVICES="${DEVICES[$i]}" "$QWEN_PYTHON" -u -m lead.inference.qwendrive_hidden_service \
        --model "$QWEN_MODEL" --socket "$socket_dir/rank$i.sock" \
        --media-root "$PWD/data/carla_leaderboard2/data" --device cuda:0 \
        --microbatch "${P8_QWEN_MICROBATCH:-8}" > "$LOGDIR/qwendrive_hidden_rank$i.log" 2>&1 &
    service_pids+=("$!")
done
for ((attempt=0; attempt<600; attempt++)); do
    ready=0
    for ((i=0; i<${#DEVICES[@]}; i++)); do
        kill -0 "${service_pids[$i]}" 2>/dev/null || { echo "Qwen rank $i failed; inspect $LOGDIR/qwendrive_hidden_rank$i.log" >&2; exit 1; }
        [[ ! -S "$socket_dir/rank$i.sock" ]] || ((ready+=1))
    done
    [[ "$ready" -ne "${#DEVICES[@]}" ]] || break
    sleep 1
done
[[ "$ready" -eq "${#DEVICES[@]}" ]] || { echo 'hidden service startup timeout' >&2; exit 1; }
else
    cp "$CACHE_DIR/metadata.json" "$LOGDIR/p8_hidden_cache_signature.json"
fi
export LEAD_TRAINING_CONFIG="logdir=$LOGDIR load_file=$POSTTRAIN \
continue_failed_training=${P8_RESUME:-false} \
image_encoder_pretrained=false use_planning_decoder=true freeze_backbone=false \
use_hidden_intent_planner=true use_semantic_intent_planner=false \
hidden_intent_socket_dir=$socket_dir qwendrive_hidden_dim=$HIDDEN_DIM hidden_intent_cache=${CACHE_DIR:-null} \
hidden_intent_frame_fraction=$FRAME_FRACTION hidden_intent_sample_seed=$SAMPLE_SEED \
p7_dataloader_workers=$WORKERS p7_intent_cache=null \
use_vlm_intent=false use_intent_decoder=false online_joint_training=false use_control_conditioning=false \
multimodal_planner=true multimodal_planner_k=6 multimodal_target_speed=true \
joint_scorer_replaces_conf=false route_select_by_conf=true \
num_route_points_prediction=10 num_route_points_smoothing=40 \
use_route_corridor_loss=false use_collision_cost=false \
use_sensor_perburtation=false use_color_aug_prob=0 \
use_discrete_command=true use_velocity=true use_rgb=true \
use_persistent_cache=false use_training_session_cache=false \
carla_num_samples=0 schedule_carla_num_samples=false batch_size=$BATCH_SIZE lr=${P8_LR:-0.0003} epochs=$EPOCHS"
LEAD_LIBNVRTC_DIR=${P8_LEAD_LIBNVRTC_DIR:-$("$LEAD_PYTHON" -c 'import pathlib, sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia/cuda_nvrtc/lib")')}
if [[ -f "$LEAD_LIBNVRTC_DIR/libnvrtc.so.12" ]]; then
    export LD_LIBRARY_PATH="$LEAD_LIBNVRTC_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
CUDA_VISIBLE_DEVICES="$GPUS" "$(dirname "$LEAD_PYTHON")/torchrun" \
    --standalone --nnodes=1 --nproc_per_node="${#DEVICES[@]}" \
    --max_restarts=0 --no-python "$LEAD_PYTHON" lead/training/train.py
