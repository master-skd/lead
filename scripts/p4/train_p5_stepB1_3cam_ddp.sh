#!/bin/bash
# P5 Step B1' (3-cam): the 3-CAMERA version of B1. Same recipe as train_p5_stepB1_ddp.sh
# (30m route, single-route output, near-weighted loss, frozen backbone, small lr,
# init from Step A), but the frozen intent head is the P6 3-camera decoder instead of
# the front-only P5a. This gives the planner that INGESTS the 3-cam intent -- the clean
# init point for B2 (K-arm planner), isolating one variable at a time:
#     Step A (P4a single-mode, 30m) -> B1' (P6 3-cam multimodal, single route)
#                                   -> B2 (P6 3-cam multimodal, K routes)
#
# vs B1 (train_p5_stepB1_ddp.sh) only 3 lines change:
#   vlm_intent_ckpt: p5a_tversky (front) -> vlm_intent_p6_3cam (3-cam)
#   vlm_cache_dir:   data/p5/vlm_cache   -> data/p6/vlm_cache_3cam
#   logdir:          p5_stepB1           -> p5_stepB1_3cam
# The VLMIntentDecoder needs NO structural change (12x36 input -> 320x384 BEV via conv +
# interpolate); training_utils re-loads the frozen head from vlm_intent_ckpt after load_file.
#
# NB closed-loop: uses 3-cam drivable features -> Qwen service must extract the FULL strip
# with the 3-cam drivable prompt (extract_vlm_p6_3cam.py), not the front crop.
#
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/p4/train_p5_stepB1_3cam_ddp.sh

source /mmu_mllm_hdd_3/liuzihan08/miniconda3/etc/profile.d/conda.sh
conda activate lead

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export LEAD_PROJECT_ROOT=/mmu_mllm_hdd_3/liuzihan08/vla/lead

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())")
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

export LEAD_TRAINING_CONFIG="logdir=outputs/local_training/p5_stepB1_3cam \
load_file=outputs/local_training/p5_stepA/model_0019.pth \
image_encoder_pretrained=false \
freeze_backbone=true \
num_route_points_smoothing=40 \
num_route_points_prediction=30 \
route_near_points=10 \
route_far_weight=0.3 \
use_vlm_intent=true \
vlm_intent_ckpt=outputs/local_training/vlm_intent_p6_3cam/model_0014.pth \
vlm_cache_dir=data/p6/vlm_cache_3cam \
vlm_manifest=data/p4/manifest.jsonl \
use_multimodal_intent=true \
use_intent_decoder=false \
use_planning_decoder=true \
use_control_conditioning=true \
use_collision_cost=true \
batch_size=128 \
lr=3e-5 \
epochs=20"

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=$nproc_per_node \
    --max_restarts=0 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --no-python \
    python3 lead/training/train.py
