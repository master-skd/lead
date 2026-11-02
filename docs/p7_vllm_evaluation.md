# P7: validate vLLM on a single RTX 3090

Run all commands from the LEAD repository root. This backend generates fresh
intents on **every frame**, with the same two-stage prompts and 512/384 token
budgets as HF. The semantic encoder, planner and candidate selection are unchanged.
This is an experimental backend: GPU startup, latency and output agreement must
be measured on the evaluation machine before a full evaluation.

## 1. Environment and standalone VLM adapter

Keep the working HF and LEAD environments. Install the vLLM backend separately:

```bash
conda create -n qwendrive_vllm python=3.12 -y
conda activate qwendrive_vllm
python -m pip install -U pip
python -m pip install 'vllm==0.27.1' 'transformers==5.14.1' accelerate
python -m pip install --no-deps -e models/Qwen-Drive-1.0-code
python -m pip install --no-deps -e scripts/p7/vllm_plugin
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch, vllm, qwen_drive; print(torch.__version__, vllm.__version__, torch.cuda.get_device_name(0))'

python scripts/p7/prepare_qwendrive_vllm.py \
  --source models/Qwen-Drive-1.0-4B \
  --output models/Qwen-Drive-1.0-4B-vllm
```

Use the actual local model directory for `--source` if different. The source
checkpoint is never modified. The adapter writes a separate approximately 9 GB
checkpoint with the `vlm.` key prefix removed; tensor bytes are copied unchanged,
without quantization. Do this once. An existing output directory is rejected.
`--inspect-only` reads headers without exporting the checkpoint.

The plugin runs the **official Qwen-Drive image processor**, rather than vLLM's
default resize path. Every request checks that expanded prompt token IDs exactly
match the official processor. A mismatch aborts the service instead of quietly
using a planner fallback. Numerical equality of model outputs is not guaranteed
across engines; the probe below compares usable planner semantics separately.

## 2. Compare the same images in HF and vLLM

Choose several existing training RGB strips (1152 × 384, left/front/right).
Include both single-branch and multiple-branch scenes. Set Python paths for this
machine; the following uses the existing evaluation-machine layout:

```bash
export HF_PY=/root/autodl-tmp/miniconda3/envs/qwendrive_official/bin/python
export VLLM_PY=/root/autodl-tmp/miniconda3/envs/qwendrive_vllm/bin/python
export QWEN_MODEL="$PWD/models/Qwen-Drive-1.0-4B"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
IMAGES=(/absolute/path/to/first.jpg /absolute/path/to/second.jpg)

CUDA_VISIBLE_DEVICES=0 "$HF_PY" scripts/p7/probe_p7_qwendrive_backend.py run \
  --backend hf --model "$QWEN_MODEL" --images "${IMAGES[@]}" \
  --output outputs/p7_backend_probe/hf.jsonl

CUDA_VISIBLE_DEVICES=0 "$VLLM_PY" scripts/p7/probe_p7_qwendrive_backend.py run \
  --backend vllm --model "$PWD/models/Qwen-Drive-1.0-4B-vllm" \
  --images "${IMAGES[@]}" --gpu-memory-utilization 0.5 \
  --output outputs/p7_backend_probe/vllm.jsonl

"$HF_PY" scripts/p7/probe_p7_qwendrive_backend.py compare \
  --hf outputs/p7_backend_probe/hf.jsonl \
  --vllm outputs/p7_backend_probe/vllm.jsonl
```

HF exits before vLLM starts, so both can use GPU 0 sequentially. Both runs default
to `follow_lane`, speed 3 m/s; use matching `--command` / `--speed` overrides for
other cases. Model loading is reported separately. JSONL records contain raw
answers, analysis/answer latency and token counts. Compare mean frame latency,
`semantic_equal_and_valid`, fallback and `hit_limit`. Inspect differing intents;
valid JSON alone is not proof of equal planner conditioning. Discard the first
frame when assessing warmed-up engine speed; the reported mean includes it.

## 3. One-route closed-loop smoke test

After the image probe, replace the checkpoint and CARLA paths below:

```bash
export P7_B2D_LEAD_PYTHON=/root/autodl-tmp/miniconda3/envs/lead/bin/python
export P7_B2D_QWEN_PYTHON=/root/autodl-tmp/miniconda3/envs/qwendrive_vllm/bin/python
export P7_B2D_QWEN_MODEL="$PWD/models/Qwen-Drive-1.0-4B-vllm"
export P7_B2D_BACKEND=vllm
export P7_B2D_VLLM_MEMORY=0.5
export CARLA_ROOT=/root/autodl-tmp/lead/3rd_party/CARLA_0915

B2D_ROUTE_IDS=3904 bash scripts/p7/eval_p7_b2d.sh \
  outputs/local_training/p7_cached_tenth/model_0019.pth "0" p7_vllm_3090_smoke
```

The 50% vLLM memory budget leaves memory for CARLA and TFv6 on the same 24 GB
GPU; it is a starting configuration, not a verified fit on the evaluation machine.
If engine graph capture exhausts memory, retry under a new tag with
`P7_B2D_VLLM_EAGER=1`; inspect the actual failure first. Keep the prompts and token
budgets unchanged for this comparison.

Service log: `outputs/p7_b2d_eval_logs/p7_vllm_3090_smoke/qwen_gpu0.log`.
Per-frame `p7_intent_trace.jsonl` files under the evaluation output record
analysis/answer timing, token metrics, fallback and selected candidate.
To revert to HF, set `P7_B2D_BACKEND=hf` and restore the HF Python and original
model paths. For the full 220 routes on one GPU, remove `B2D_ROUTE_IDS` and use a
new tag with the same `"0"` GPU argument.
