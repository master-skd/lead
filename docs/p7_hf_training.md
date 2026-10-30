# P7: generate structured intents once, then train the planner

The main P7 experiment has two stages. Frozen Qwen-Drive generates per-frame
semantic JSON once. The `lead` environment then trains SemanticIntentEncoder,
K paired routes/speeds and route confidence from LEAD posttrain weights, reusing
the saved JSON for every epoch. There is no independent intent mask decoder,
anchor cache, hidden-state cache or joint scorer. Generation uses the reviewed
three-camera order and the same two-stage HF prompt as the online baseline.

From the LEAD project root, prepare the `lead` and GPU `qwendrive_official`
environments described in the [README](../README.md). Put the posttrain
`model_0030.pth` and `config.json` together in
`outputs/local_training/posttrain/`.

## Small end-to-end check

Use separate manifest/cache paths for the small check so it cannot be mistaken
for a full-data generation run:

```bash
CONDA_BASE=$(conda info --base)
export P7_LEAD_PYTHON="$CONDA_BASE/envs/lead/bin/python"
export P7_QWEN_PYTHON="$CONDA_BASE/envs/qwendrive_official/bin/python"
export P7_QWEN_MODEL="$PWD/models/Qwen-Drive-1.0-4B"

CUDA_VISIBLE_DEVICES=0 P7_GENERATE_FRAMES=32 \
P7_MANIFEST=data/p7/smoke_manifest.jsonl P7_INTENT_DIR=data/p7/smoke_intents \
  bash scripts/p7/generate_p7_intents.sh

CUDA_VISIBLE_DEVICES=0 P7_BATCH_SIZE=4 P7_EPOCHS=1 \
P7_MANIFEST=data/p7/smoke_manifest.jsonl \
P7_INTENT_CACHE=data/p7/smoke_intents/intents.sqlite \
P7_LOGDIR=outputs/local_training/p7_cached_smoke \
  bash scripts/p7/train_p7_cached.sh
```

The training dataset explicitly selects the manifest's frames. It does not
independently draw 32 random frames that might be absent from the cache.

## Full eight-GPU run

With the Python/model variables above set, run generation once:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/p7/generate_p7_intents.sh
```

Defaults are all posttrain bucket frames, analysis limit 512 tokens and answer
limit 384 tokens. Generation exports `data/p7/intent_manifest.jsonl` using the
same first navigation command and current speed as CARLAData, then launches
one independent HF worker per GPU. Each worker writes an append-only
`data/p7/intent_cache/shard-NNNNN.jsonl` and a `generate_rankN.log`. Records
contain a portable scenario/route/rgb/frame key, an absolute image path for
review, navigation, speed, analysis text, raw answer, parsed JSON, generation
errors and pre-fallback usable branch counts.

Re-run the same command to resume. Completed records are skipped and an
interrupted final JSONL write is repaired. Use the same manifest, number of
shards, model config and prompt/token settings; changed generation signatures
are rejected. `P7_RETRY_FAILED=1` retries frames with no usable branch, appending
new records; the latest record is used in the index. Use a new output directory
when deliberately changing generation settings. Full generation still incurs
HF autoregressive inference once per frame; this change eliminates repeated
generation during subsequent epochs and experiments.

After all workers finish, the launcher verifies full manifest coverage and
atomically creates `intents.sqlite`. The JSONL remains available for review;
SQLite stores the JSON records with an index, avoiding a million small files
or a full in-memory JSON dictionary during training. Read
`intents.summary.json` before the full training run: it reports raw and usable
branch histograms, fallback counts and parser/navigation rejection reasons.
The usable histogram is counted **before** inserting fallback slot 0.

Then train, with no Qwen-Drive process or weights loaded:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
P7_BATCH_SIZE=64 P7_WORKERS=4 P7_EPOCHS=20 \
P7_LOGDIR=outputs/local_training/p7_cached_full \
  bash scripts/p7/train_p7_cached.sh
```

Batch size is global. The cache and its generation manifest must both be
available; `P7_INTENT_CACHE` and `P7_MANIFEST` override their default paths.
Copy both to another machine: lookup uses portable frame keys, not the stored
absolute review path. Missing frames, mismatched navigation/speed and a
mismatched manifest fingerprint stop training rather than silently becoming
fallback samples. No pretrain checkpoint or optimizer state is required for a
new experiment initialized from posttrain.

The planner architecture and losses are unchanged: K=6 capacity, ten route
points, paired K speeds, single temporal-waypoint auxiliary head, masked WTA
and route confidence. Existing posttrain route queries are copied to all K
slots. Check the `P7 checkpoint load ...` lines. Every 50 steps rank 0 logs
valid slots, fallback count and a real branch histogram with fallback counted
as zero. Ten consecutive entirely fallback batches still stop training.
Timing includes `intent_cache_read` and `qwendrive_parse`.

CPU tests cover online/offline tensor parity, data-root relocation, incomplete
cache rejection, interrupted writes and generation resume. A real GPU
end-to-end cached training run still needs validation on the training machine.

## Optional online comparison

`scripts/p7/train_p7_hf_online.sh` still runs frozen HF generation during each
training step when `P7_INTENT_CACHE` is unset. It requires both environments
and starts one Qwen service per GPU. `P7_INTENT_PREFETCH=1` (default) overlaps
next-batch generation with the current planner step; setting it to 0 runs
synchronously. This mode repeats VQA every epoch. Its timing phases are
`qwendrive_wait` (or synchronous `qwendrive_intent`), `qwendrive_parse` and
`next_batch_prefetch`; errors are in `qwendrive_rankN.log`.
