# P8 Bench2Drive closed-loop evaluation

P8 runs **one frozen Qwen-Drive forward on each model inference step**. It does not
read the training hidden-state cache or generate AR analysis/JSON. The live RGB
strip follows the same quality-30 JPEG → PIL RGB boundary as training. Qwen reads
front/front-left/front-right views, navigation and signed ego speed through the same
processor/prompt used for cache generation. Image and text tokens, masks and view
IDs are sent in memory with request/frame/timestamp/command/speed alignment checks.

The trained HiddenIntentEncoder produces K latent modes with cross-attention.
Their gated residuals condition the inherited TFv6 route queries. All K modes are
candidates; **confidence argmax selects one index for both route and speed**. No
corridor, anchor, AR semantic parser or DrivoR scorer participates. Temporal waypoints
remain the shared auxiliary output. Existing LEAD speed decoding/braking and PID
controllers execute the selected pair.

## Files needed on the evaluation machine

Copy these from `outputs/local_training/p8_hidden_cached_tenth/` into one directory:

- `model_0019.pth`
- `config.json`
- `p8_hidden_cache_signature.json` (small provenance metadata)

Optimizer/scheduler/scaler and the approximately 230 GB feature cache are not
required. Prepare the original Qwen-Drive model directory and install its separate
source package in the Qwen Python environment. The LEAD and Qwen environments may
be separate processes; Qwen needs the same feature extraction dependencies used
for training. Inspect the saved signature's `runtime` field to check versions.

When the signature is present next to the checkpoint, startup checks Qwen weights,
config, tokenizer/processor assets, source, feature extraction code, dimensions and
Transformers/torchvision/Pillow versions against training. Differences abort before
model inference. Other runtime differences are reported, not treated as proof of
numerical parity. Without the sidecar, the launcher reports that provenance was not
verified. `P8_B2D_SIGNATURE=/path/to/p8_hidden_cache_signature.json` supplies it
explicitly if stored elsewhere.

## Single route first

From the evaluation machine's LEAD root, replace environment/model/CARLA paths:

```bash
export P8_B2D_LEAD_PYTHON=/path/to/miniconda3/envs/lead/bin/python
export P8_B2D_QWEN_PYTHON=/path/to/miniconda3/envs/qwendrive/bin/python
export P8_B2D_QWEN_MODEL="$PWD/models/Qwen-Drive-1.0-4B"
export CARLA_ROOT=/path/to/CARLA_0915

B2D_ROUTE_IDS=3904 bash scripts/p8/eval_p8_b2d.sh \
  outputs/local_training/p8_hidden_cached_tenth/model_0019.pth "0" p8_hidden_smoke
```

On the prepared nlp_hdd checkout the default paths point to its local `lead` and
`qwendrive` environments and model. Use GPU `"0"` for a single 3090. CARLA, Qwen
and TFv6 share this GPU; the model service runs with microbatch 1 and no decode KV
cache. Verify actual peak memory and latency on that machine.

`P8_B2D_PREFLIGHT_ONLY=1` checks checkpoint configuration/path setup and exposes only
one checkpoint without launching CARLA/Qwen. The full service performs the expensive
weight fingerprint check after that. `P8_B2D_SERVICE_WAIT` defaults to 600 seconds;
`P8_B2D_AGENT_TIMEOUT` defaults to 600, and `P8_B2D_ROUTE_TIMEOUT` to 86400.
Checkpoint weights load strictly; optional legacy velocity/safety scorers are rejected
for this baseline.

## Full 220 routes

```bash
unset B2D_ROUTE_IDS
bash scripts/p8/eval_p8_b2d.sh \
  outputs/local_training/p8_hidden_cached_tenth/model_0019.pth "0" p8_hidden_full
```

For multiple GPUs, use `"0 1 2 3 4 5 6 7"`. Each GPU has its own Qwen service,
CARLA process and route partition. Full-run result files are validated and then
merged under `outputs/b2d_p8_hidden_full/`. A subset run is not a 220-route score.

Use the same tag to resume. Before restarting, malformed/empty endpoints and agent
or simulation crashes are archived and retried. Genuine driving failures (for
example, blocked or deviated routes) remain scored results. Failed infrastructure
runs are not silently merged as completed evaluations.

## Diagnostics

- Qwen log: `outputs/p8_b2d_eval_logs/TAG/qwen_gpu0.log`.
- CARLA/evaluator logs: the shared driver's `/tmp/b2d_TAG/` directory.
- Route outputs: `outputs/local_evaluation_TAG/ROUTE_ID/`.
- `p8_hidden_trace.jsonl`: aligned input frame/timestamp, token counts, feature
  timings, candidate routes/speeds, confidence logits, selected index and selected
  raw model route/speed. Each prediction verifies that both outputs belong to the
  confidence-selected mode. These raw predictions precede controller braking/PID.

Feature failures abort the route rather than substitute empty tokens or an
untrained LEAD fallback. CPU integration/preflight tests cover the handoff and
pairing; a real CARLA/GPU closed-loop run is still required to establish performance.
