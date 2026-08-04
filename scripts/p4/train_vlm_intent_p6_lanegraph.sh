#!/bin/bash
# P6 verification: train a VLMIntentDecoder on the 3-CAMERA VLM features (12x36 wide,
# ~4000-frame subset) to test whether SIDE arms (left/right turns) -- invisible to the
# front-only P5a -- can be lit in the BEV intent. Same decoder/target as P5a, only the
# cache changes to data/p6/vlm_cache_3cam. Verification-scale (small subset), not final.
#
# The decoder needs NO structural change: it takes (B,12,36,2560), its convs are
# space-agnostic, and the final F.interpolate resizes to the fixed (1,320,384) BEV grid.
#
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/p4/train_vlm_intent_p6_3cam.sh

source /mmu_mllm_hdd_3/liuzihan08/miniconda3/etc/profile.d/conda.sh
conda activate lead

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export LEAD_PROJECT_ROOT=/mmu_mllm_hdd_3/liuzihan08/vla/lead
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())")
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=$nproc_per_node \
    --max_restarts=0 \
    scripts/p4/train_vlm_intent.py \
    --vlm-cache data/p6/vlm_cache_3cam \
    --manifest data/p4/manifest.jsonl \
    --logdir outputs/local_training/vlm_intent_p6_lanegraph \
    --multimodal-intent \
    --tversky-weight 0.75 \
    --batch-size 256 \
    --lr 6e-4 \
    --epochs 15 \
    --num-workers 6 \
    --lanegraph-label-dir data/p6/lanegraph_label \
    "$@"
