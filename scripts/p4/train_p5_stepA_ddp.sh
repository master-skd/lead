#!/bin/bash
# P5 Step A: extend the route head from ~10m (10 pts) to ~30m (30 pts), STILL SINGLE-MODE.
# Goal: prove the 30m route head is a stable foundation before layering multimodality
# (Step B). This is the geometry the K arms need -- junction forks live at 15-25m, which
# the old 10m head could never see, so multimodal WTA on a 10m head would collapse.
#
# Critical differences from the P5b recipe that COLLAPSED (S=72.9 / M=66.2):
#   - load_file = P2 (control_p2_full), NOT pretrain. The planner INHERITS P2's converged
#     weights instead of training from random. P5b's random planner on the 1/17 subset was
#     the collapse cause (open-loop wp_ade 0.416 vs P2 0.178, zero-intent == real-intent).
#   - freeze_backbone=true: only the planner increment is finetuned.
#   - The 20 newly-added far-range route queries are the ONLY random increment; the near 10
#     route queries + waypoint/speed queries are migrated from P2 verbatim
#     (see training_utils._migrate_planner_query).
#   - small lr (finetune magnitude, not the 3e-4 from-scratch magnitude).
#
# Intent path is the PROVEN P4a+P2 closed-loop winner (94.1 DS): command-conditioned
# single-mode VLM intent, frozen. Step A changes ONLY the route length.
#
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/p4/train_p5_stepA_ddp.sh

source /mmu_mllm_hdd_3/liuzihan08/miniconda3/etc/profile.d/conda.sh
conda activate lead

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())")
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

export LEAD_TRAINING_CONFIG="logdir=outputs/local_training/p5_stepA \
load_file=outputs/local_training/control_p2_full/model_0030.pth \
image_encoder_pretrained=false \
freeze_backbone=true \
num_route_points_smoothing=40 \
num_route_points_prediction=30 \
route_near_points=10 \
route_far_weight=0.3 \
use_vlm_intent=true \
vlm_intent_ckpt=outputs/local_training/vlm_intent_p4a/model_0014.pth \
vlm_cache_dir=data/p4/vlm_cache \
vlm_manifest=data/p4/manifest.jsonl \
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
