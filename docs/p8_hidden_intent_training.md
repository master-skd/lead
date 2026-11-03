# P8: Qwen-Drive hidden tokens → continuous intent → TFv6

This implements the main architecture in `QwenDrive_HiddenState_Intent_Planner_Design.md`.
P7 AR and P6 corridor modes remain available. P8 is an independent, mutually
exclusive mode (`use_hidden_intent_planner=true`). GPU latency and planning quality
are unverified until the probes/training below run on the target GPU.

## Architecture and supervision

1. Read the current original three-camera training JPEG. The live agent reproduces
   the quality-30 JPEG RGB path, then transports it losslessly. The official
   Qwen-Drive processor orders views **front, front-left, front-right**.
2. Encode a fixed short planning prompt containing current navigation and speed.
   A frozen, eval-mode Qwen-Drive VLM makes one forward per sample (batched). P8
   calls the base multimodal model, not `generate` or the vocabulary projection.
   The separate Planning Expert does not participate.
3. Select **all final normalized image and text tokens**. Retaining text allows
   the scene representation to read navigation after the image tokens in a causal
   model. Do not reshape using old Qwen3-VL's hardcoded spatial dimensions.
4. Train a LayerNorm/Linear/LayerNorm adapter, view embeddings, explicit navigation
   and speed conditioning, and K=6 learned mode queries. Cross-attention reads the
   shared scene memory; a residual FFN produces `[B,K,D]` latent intents. There is
   no initial token compression, maneuver classifier, corridor, threshold or anchor.
5. Use the existing sigmoid gate (initial value 0.1) and linear projection to add
   each latent intent to its mode's route queries. Replicate posttrain's original
   route queries K times and preserve the waypoint/global speed query tail. Mode
   queries break symmetry; no extra route mode embedding is added in this version.
6. Keep K routes, K paired speed distributions, route confidence, and shared temporal
   waypoint prediction. Geometry WTA assigns the same winner to route and speed
   supervision; confidence uses the existing winner-vs-other BCE. Temporal targets
   retain the existing supervision. All K latent modes are candidates, **not**
   verified road branches. Inference selects confidence argmax and gathers route
   and speed with the same index.

The Qwen service freezes every Qwen parameter. TFv6/backbone, adapter, cross-attention,
mode queries, gate and prediction heads train through existing LEAD losses.
AR teacher supervision is **not implemented in this first baseline**; no AR JSON or
lane graph is required. The preferred workflow below caches the frozen Qwen tokens
once and reuses them across planner epochs. The online baseline remains available. There is no explicit diversity loss.
Multiple candidates may collapse; this is measured rather than assumed solved.

## Environments and single-forward probe

Reuse the existing `lead` and `qwendrive_official` environments. Model source code
is the separately installed official Qwen-Drive repository. Original model weights
are used unchanged; the P7 vLLM adapter export is not needed.

From the LEAD repository root, set paths to your machine:

```bash
export P8_LEAD_PYTHON=/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead/bin/python
export P8_QWEN_PYTHON=/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/qwendrive_official/bin/python
export P8_QWEN_MODEL=/mmu_mllm_hdd_3/liuzihan08/models/Qwen-Drive-1.0-4B
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES=0 "$P8_QWEN_PYTHON" scripts/p8/probe_qwendrive_hidden.py \
  --model "$P8_QWEN_MODEL" \
  --images /absolute/path/to/three_camera_strip.jpg \
  --repeats 6 --microbatch 1
```

The report `outputs/p8_hidden_probe.json` contains actual hidden dimensions, token
counts, preprocessing/H2D and forward/D2H wall times, warmed-up p50/p95, and peak
allocated GPU memory. CUDA stages synchronize before timing. These timings include
feature conversion to CPU; they do not include socket transfer, TFv6 or CARLA.
Use multiple `--images` and `--microbatch 8` to measure batched extraction.
`--check-batch-parity` additionally compares batched and single-sample features.
Do this before a large run. A forward-only architecture still needs GPU benchmarking.

## Existing nlp_hdd checkout

For the prepared copy at `/mmu_nlp_hdd/liuzihan08/lead`, use:

```bash
cd /mmu_nlp_hdd/liuzihan08/lead
bash scripts/p8/run_p8_nlp.sh extract
# After extraction finishes:
bash scripts/p8/run_p8_nlp.sh train
```

This wrapper selects the local `lead` environment and
`/mmu_nlp_hdd/liuzihan08/miniconda3/envs/qwendrive/bin/python`, with the model at
`/mmu_nlp_hdd/liuzihan08/models/Qwen-Drive-1.0-4B`. The Qwen environment is an isolated
venv that reuses CUDA 12.9 PyTorch from the existing `vllm_qwen35_cu129` environment,
with Transformers 5.14.1 and the local editable Qwen-Drive source installation.
Keep that base environment in place. All runtime environment/model paths are on
`nlp_hdd`. Override the same `P8_*` batch/worker/cache/log variables as below.

## Preferred: cache the fixed tenth once, then train the planner

After setting the three Python/model variables above, run from the LEAD root:

```bash
# Stage 1: export the fixed 1/10 manifest and extract FP16 token shards on 8 GPUs.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
P8_POSTTRAIN=outputs/local_training/posttrain/model_0030.pth \
P8_HIDDEN_CACHE=data/p8/hidden_cache \
P8_FRAME_FRACTION=0.1 P8_SAMPLE_SEED=2026 \
P8_QWEN_MICROBATCH=32 P8_CACHE_WORKERS=16 \
bash scripts/p8/extract_p8_hidden_cache.sh

# Stage 2: train HiddenIntentEncoder and TFv6; no Qwen process is started.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
P8_POSTTRAIN=outputs/local_training/posttrain/model_0030.pth \
P8_HIDDEN_CACHE=data/p8/hidden_cache \
P8_BATCH_SIZE=64 P8_WORKERS=8 P8_EPOCHS=20 \
bash scripts/p8/train_p8_cached.sh
```

Stage 1 uses the Qwen environment; stage 2 needs only the LEAD environment,
posttrain checkpoint and original CARLA inputs/labels. The cached training script
defaults to `outputs/local_training/p8_hidden_cached_tenth`. `P8_HIDDEN_PREFETCH=1`
is enabled for cached training to overlap one batch of cache reading with planner
training. `P8_PREFLIGHT_ONLY=1` validates cache metadata/paths without starting GPUs.

Extraction writes `data/p8/hidden_manifest.jsonl` (portable frame keys, navigation
and speed) and its sampling metadata. The cache directory contains:

- `metadata.json`: manifest, model/processor/source fingerprints, package versions,
  final layer, dtype and sampling settings. Weight SHA256 is checked during prepare
  and generator startup; this scan can take time before feature extraction begins.
- `manifest.jsonl` and `manifest.meta.json`: the exact extracted frame set.
- `shard-00000.bin` / `.jsonl`, etc.: packed raw FP16 hidden tokens and int8 view IDs,
  plus per-frame offsets, navigation/speed and payload checksums.
- `frames.sqlite`: complete lookup index, built only after every manifest frame is
  present. `generate_rankN.log` records extraction progress and batch timing.

Tokens are not pooled or quantized further. Padding is removed before storage and
reconstructed on batch read, preserving all image/text tokens and their masks.
At roughly 474 tokens × 2560 dimensions × FP16, each frame is about 2.4 MB;
95,000 frames require roughly **230 GB** of feature storage. Actual lengths and
storage depend on the inputs. Eight packed shards avoid tens of thousands of small
feature files. Original images/labels remain necessary for LEAD training.

Rerun the same extraction command to resume; completed frames are skipped and only
uncommitted tails are repaired. Keep the original GPU/shard count when resuming;
the extractor halves its batch automatically on CUDA OOM and retries the same
unwritten frames. Defaults are batch 32 and 16 image-loading threads per GPU.
You can override `P8_QWEN_MICROBATCH` / `P8_CACHE_WORKERS` when launching. Do not change the model, prompt, environment
or manifest midway through a cache. Concurrent writers to the same shard are rejected.
If extraction finished but indexing was interrupted, run:

```bash
"$P8_LEAD_PYTHON" scripts/p8/cache_qwendrive_hidden.py index \
  --cache-dir data/p8/hidden_cache
```

Training uses **exactly the cached manifest**, without sampling another tenth.
Missing cached frames, stale indexes, corrupt payloads or mismatched navigation/speed
stop the run. The log directory records the cache signature and selected frame list.
Move the entire cache directory together; keys and packed-file references are portable
across disk prefixes. This cache accelerates training only; closed loop still uses
fresh Qwen forward features per frame.

## Optional online baseline: train from posttrain with a fixed tenth of frames

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
P8_POSTTRAIN=outputs/local_training/posttrain/model_0030.pth \
P8_LOGDIR=outputs/local_training/p8_hidden_tenth \
P8_BATCH_SIZE=64 P8_WORKERS=8 P8_QWEN_MICROBATCH=8 \
P8_EPOCHS=20 P8_FRAME_FRACTION=0.1 \
bash scripts/p8/train_p8_hidden.sh
```

The checkpoint needs its sibling `config.json`. The script uses 10 route points,
K=6, K speeds and confidence. Posttrain route query length must match; P8 refuses
an incompatible source instead of silently retaining randomly initialized queries.
Standard checkpoint loading logs module match counts. Check these before continuing.

Sampling starts from all frames in the active posttrain buckets at epoch 0,
without bucket resampling/repetition. Unique portable frame keys are sorted by
SHA256(seed + frame key); the first `floor(N*0.1)` keys are used. The default seed
is 2026. This is frame sampling, **not a route-separated validation split**. The
selected frame set is fixed across epochs, DDP ranks and disk prefixes, while its
order shuffles normally. Changing fraction/seed or the bucket pool changes the
experiment. `p8_train_frames.txt` and `p8_subset.json` in the training log directory
record the selected keys and their SHA256. It does not necessarily match the earlier P7 cache's sampled set.

Global batch is `P8_BATCH_SIZE`; per-GPU planner batch is global batch / GPU count.
`P8_QWEN_MICROBATCH` limits service forward memory independently. Lower it on OOM.
`P8_WORKERS` caps DataLoader workers per training rank. Sensor perturbations and
color augmentation are disabled to keep the original images/labels aligned.
Qwen hidden dimension is read from the model config and checked against responses.
CARLA speed records can be negative (including small standstill values). The prompt
preserves finite signed values consistently in cache, online training and live inference;
NaN/Inf are rejected.

The first run is synchronous. Set `P8_HIDDEN_PREFETCH=1` for optional one-batch
lookahead after validation: batch t+1 extraction overlaps batch t training, but
responses remain attached to their original batch. Socket replies echo request ID,
frame ID, navigation, speed, and timestamp (unknown for training JPEGs). A mismatch,
missing image, service error or empty feature sequence stops the run. Feature arrays
are transmitted as an in-memory binary buffer; no feature files are written.

Logs include Qwen preprocessing/forward timing, hidden token counts, gate value,
semantic residual/query norm ratio, intent-adapter gradient norm, candidate pair
distance, near-duplicate fraction (<0.1 m mean point distance), and WTA winner
frequencies. The shared training timing recorder remains available. There is no
new held-out evaluator in this initial baseline; training losses are not evidence
of improved planning quality.

To resume, use the same settings/log directory and `P8_RESUME=true`, with
`P8_POSTTRAIN` pointing to the completed P8 `model_NNNN.pth`. Keep optimizer,
scheduler and scaler files from that epoch for the existing LEAD resume path.

## Closed loop after training

See the [P8 B2D evaluation guide](p8_b2d_evaluation.md) for checkpoint transfer,
feature-contract checks, single-GPU/full evaluation, resume and diagnostics.

```bash
export P8_B2D_LEAD_PYTHON="$P8_LEAD_PYTHON"
export P8_B2D_QWEN_PYTHON="$P8_QWEN_PYTHON"
export P8_B2D_QWEN_MODEL="$P8_QWEN_MODEL"
export CARLA_ROOT=/path/to/CARLA_0915
B2D_ROUTE_IDS=3904 bash scripts/p8/eval_p8_b2d.sh \
  outputs/local_training/p8_hidden_cached_tenth/model_0019.pth "0" p8_hidden_smoke
```

The agent obtains fresh same-frame tokens on every model step and does no AR
generation. Per-frame `p8_hidden_trace.jsonl` records feature/frame metadata and
paired selection. Stop other Qwen services first. CARLA, Qwen and TFv6 share memory
on the selected GPU; a successful standalone probe does not establish a memory
budget or latency for their combined execution. Remove `B2D_ROUTE_IDS` and use a
new tag for a full evaluation only after the one-route smoke test succeeds.
