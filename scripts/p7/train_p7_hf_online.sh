#!/usr/bin/env bash
# Train P7 from LEAD posttrain, with frozen Qwen-Drive HF VQA per training rank.
set -euo pipefail
cd "$(dirname "$0")/../.."

LEAD_PYTHON=${P7_LEAD_PYTHON:-/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead/bin/python}
QWEN_PYTHON=${P7_QWEN_PYTHON:-/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/qwendrive_official/bin/python}
QWEN_MODEL=${P7_QWEN_MODEL:-/mmu_mllm_hdd_3/liuzihan08/models/Qwen-Drive-1.0-4B}
POSTTRAIN=${P7_POSTTRAIN:-outputs/local_training/posttrain/model_0030.pth}
LOGDIR=${P7_LOGDIR:-outputs/local_training/p7_hf_online}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
BATCH_SIZE=${P7_BATCH_SIZE:-64}
WORKERS=${P7_WORKERS:-4}
EPOCHS=${P7_EPOCHS:-20}
INTENT_CACHE=${P7_INTENT_CACHE:-}
IFS=',' read -r -a DEVICES <<< "$GPUS"

for value in "$BATCH_SIZE" "$WORKERS" "$EPOCHS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "batch/workers/epochs must be positive integers" >&2; exit 1; }
done
(( ${#DEVICES[@]} > 0 && BATCH_SIZE % ${#DEVICES[@]} == 0 )) || {
    echo "P7_BATCH_SIZE must be divisible by the GPU count" >&2; exit 1;
}
[[ -x "$LEAD_PYTHON" \
   && -f "$POSTTRAIN" && -f "$(dirname "$POSTTRAIN")/config.json" \
   && -d data/carla_leaderboard2/data && -d data/carla_leaderboard2/buckets ]] || {
    echo "P7 preflight failed: check LEAD Python, posttrain and CARLA data" >&2; exit 1;
}
if [[ -n "$INTENT_CACHE" ]]; then
    [[ -f "$INTENT_CACHE" ]] || { echo "Missing P7 intent cache: $INTENT_CACHE" >&2; exit 1; }
    [[ -f "${P7_MANIFEST:-}" ]] || { echo "Set P7_MANIFEST to the cache's generation manifest" >&2; exit 1; }
else
    [[ -x "$QWEN_PYTHON" && -f "$QWEN_MODEL/model.safetensors" ]] || {
        echo "Missing Qwen environment or model" >&2; exit 1;
    }
fi
mkdir -p "$LOGDIR"
socket_dir=$(mktemp -d "${TMPDIR:-/tmp}/lead_p7_intent.XXXXXX")
service_pids=()
cleanup() {
    for pid in "${service_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${service_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    for ((i=0; i<${#DEVICES[@]}; i++)); do rm -f "$socket_dir/rank$i.sock"; done
    rmdir "$socket_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export P7_INTENT_PREFETCH=${P7_INTENT_PREFETCH:-1}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "$INTENT_CACHE" ]]; then
for ((i=0; i<${#DEVICES[@]}; i++)); do
    CUDA_VISIBLE_DEVICES="${DEVICES[$i]}" "$QWEN_PYTHON" -u -m lead.inference.qwendrive_intent_service \
        --model "$QWEN_MODEL" --socket "$socket_dir/rank$i.sock" \
        --media-root "$PWD/data/carla_leaderboard2/data" --device cuda:0 \
        --analysis-tokens "${P7_ANALYSIS_TOKENS:-512}" \
        --answer-tokens "${P7_ANSWER_TOKENS:-384}" \
        > "$LOGDIR/qwendrive_rank$i.log" 2>&1 &
    service_pids+=("$!")
done
ready=0
for ((attempt=0; attempt<600; attempt++)); do
    ready=0
    for ((i=0; i<${#DEVICES[@]}; i++)); do
        kill -0 "${service_pids[$i]}" 2>/dev/null || {
            echo "Qwen-Drive rank $i exited; inspect $LOGDIR/qwendrive_rank$i.log" >&2; exit 1;
        }
        [[ -S "$socket_dir/rank$i.sock" ]] && ((ready+=1)) || true
    done
    [[ "$ready" -eq "${#DEVICES[@]}" ]] && break
    sleep 1
done
[[ "$ready" -eq "${#DEVICES[@]}" ]] || { echo "Qwen-Drive service startup timed out" >&2; exit 1; }
fi

export LEAD_TRAINING_CONFIG="logdir=$LOGDIR load_file=$POSTTRAIN \
image_encoder_pretrained=false use_planning_decoder=true \
use_semantic_intent_planner=true p7_intent_socket_dir=$socket_dir \
p7_intent_cache=${INTENT_CACHE:-null} \
p7_intent_manifest=${P7_MANIFEST:-null} \
p7_dataloader_workers=$WORKERS use_vlm_intent=false use_intent_decoder=false \
online_joint_training=false use_control_conditioning=false \
multimodal_planner=true multimodal_planner_k=6 multimodal_target_speed=true \
joint_scorer_replaces_conf=false route_select_by_conf=true \
num_route_points_prediction=10 num_route_points_smoothing=40 \
use_route_corridor_loss=false use_collision_cost=false \
use_sensor_perburtation=false use_color_aug_prob=0 \
use_discrete_command=true use_velocity=true use_rgb=true \
use_persistent_cache=false use_training_session_cache=false \
carla_num_samples=${P7_TRAIN_FRAMES:-0} batch_size=$BATCH_SIZE \
lr=${P7_LR:-0.0003} epochs=$EPOCHS"

LEAD_LIBNVRTC_DIR=${P7_LEAD_LIBNVRTC_DIR:-$("$LEAD_PYTHON" -c 'import pathlib, sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia/cuda_nvrtc/lib")')}
if [[ -f "$LEAD_LIBNVRTC_DIR/libnvrtc.so.12" ]]; then
    export LD_LIBRARY_PATH="$LEAD_LIBNVRTC_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
CUDA_VISIBLE_DEVICES="$GPUS" "$(dirname "$LEAD_PYTHON")/torchrun" \
    --standalone --nnodes=1 --nproc_per_node="${#DEVICES[@]}" \
    --max_restarts=0 --no-python "$LEAD_PYTHON" lead/training/train.py
