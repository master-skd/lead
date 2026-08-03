#!/bin/bash
# P5 B2: multimodal planner -- emits K route arms + per-mode confidence, anchor-conditioned
# queries, WTA loss. Inits from B1' (P6 3-cam single-route planner), replicating its route
# queries K times (anchors differentiate them). Only the multimodal increment + conf head
# are new; backbone + intent head frozen, small lr.
#   Step A -> B1 -> B1' (P6 3-cam single) -> B2 (K arms)  <- here
#
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/p4/train_p5_stepB2_ddp.sh

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

export LEAD_TRAINING_CONFIG="logdir=outputs/local_training/p5_stepB2 \
load_file=outputs/local_training/p5_stepB1_3cam/model_0019.pth \
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
multimodal_planner=true \
multimodal_planner_k=6 \
anchor_cache_dir=data/p6/anchor_cache \
vlm_3cam=true \
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
