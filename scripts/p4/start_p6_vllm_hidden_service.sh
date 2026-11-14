#!/usr/bin/env bash
set -euo pipefail

# Local vLLM hidden-state service used by the optional P6 vLLM training backend.
P6_VLLM_BIN=${P6_VLLM_BIN:-vllm}
P6_VLLM_PYTHON=${P6_VLLM_PYTHON:-$(dirname "$(command -v "$P6_VLLM_BIN")")/python}
P6_VLLM_MODEL=${P6_VLLM_MODEL:?set P6_VLLM_MODEL to Qwen3-VL-4B-Instruct}
P6_VLLM_MEDIA_ROOT=${P6_VLLM_MEDIA_ROOT:?set P6_VLLM_MEDIA_ROOT to CARLA RGB root}
P6_VLLM_SHARED_DIR=${P6_VLLM_SHARED_DIR:-/dev/shm/p6_vllm_hidden_pilot}
P6_VLLM_PORT=${P6_VLLM_PORT:-8007}
P6_VLLM_KV_PORT=${P6_VLLM_KV_PORT:-14579}
P6_VLLM_MAX_MODEL_LEN=${P6_VLLM_MAX_MODEL_LEN:-1024}
P6_VLLM_MAX_NUM_SEQS=${P6_VLLM_MAX_NUM_SEQS:-16}
P6_VLLM_GPU_MEMORY_UTILIZATION=${P6_VLLM_GPU_MEMORY_UTILIZATION:-0.6}
# The model's default 16,777,216-pixel image ceiling makes vLLM profile a
# 16K-token dummy image, although P6's real panoramas are 384x1152 (442,368 px).
P6_VLLM_MAX_PIXELS=${P6_VLLM_MAX_PIXELS:-524288}

if [[ ! -f "$P6_VLLM_MODEL/config.json" || ! -d "$P6_VLLM_MEDIA_ROOT" ]]; then
  echo "model config or CARLA RGB root missing" >&2
  exit 1
fi
mkdir -p -m 700 "$P6_VLLM_SHARED_DIR"

P6_VLLM_LAYERS=$(
  P6_VLLM_MODEL="$P6_VLLM_MODEL" "$P6_VLLM_PYTHON" -c \
    'import json, os, pathlib; print(json.loads((pathlib.Path(os.environ["P6_VLLM_MODEL"]) / "config.json").read_text())["text_config"]["num_hidden_layers"])'
)
P6_VLLM_SPEC=$(printf '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[%s]}}}' "$P6_VLLM_LAYERS")
P6_VLLM_KV=$(P6_VLLM_SHARED_DIR="$P6_VLLM_SHARED_DIR" P6_VLLM_KV_PORT="$P6_VLLM_KV_PORT" "$P6_VLLM_PYTHON" -c \
  'import json, os; print(json.dumps({"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_port":int(os.environ["P6_VLLM_KV_PORT"]),"kv_connector_extra_config":{"shared_storage_path":os.environ["P6_VLLM_SHARED_DIR"]}}))')
P6_VLLM_MM=$(P6_VLLM_MAX_PIXELS="$P6_VLLM_MAX_PIXELS" "$P6_VLLM_PYTHON" -c \
  'import json, os; print(json.dumps({"max_pixels":int(os.environ["P6_VLLM_MAX_PIXELS"])}))')

exec "$P6_VLLM_BIN" serve "$P6_VLLM_MODEL" \
  --served-model-name p6-vllm-pilot \
  --host 127.0.0.1 --port "$P6_VLLM_PORT" \
  --allowed-local-media-path "$P6_VLLM_MEDIA_ROOT" \
  --mm-processor-kwargs "$P6_VLLM_MM" \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --max-model-len "$P6_VLLM_MAX_MODEL_LEN" \
  --max-num-seqs "$P6_VLLM_MAX_NUM_SEQS" \
  --gpu-memory-utilization "$P6_VLLM_GPU_MEMORY_UTILIZATION" \
  --speculative-config "$P6_VLLM_SPEC" \
  --kv-transfer-config "$P6_VLLM_KV" \
  "$@"
