# P7 HF intent-conditioned planner training

This is the first executable P7 training path. Each GPU has a frozen
Qwen-Drive-1.0 HF VQA service in the `qwendrive_official` environment and a LEAD
training rank in the `lead` environment. The service reads the original CARLA
three-camera JPEG, crops front/front-left/front-right in the reviewed order,
runs the same two-stage rule-intent prompt as the CPU probe, and returns text.
The training rank parses it into fixed semantic slots and trains K paired routes
and speeds with masked WTA and route confidence. It does not load a Planning
Expert checkpoint, the old Qwen3-VL hidden-state head, anchor cache, corridor
conditioning or the joint scorer. The Qwen-Drive checkpoint is read-only.

On a GPU machine with all CARLA sensors, bucket collection and the posttrain
checkpoint present, first run a small end-to-end smoke test:

```bash
cd /mmu_mllm_hdd_3/liuzihan08/vla/lead
CUDA_VISIBLE_DEVICES=0 P7_BATCH_SIZE=4 P7_TRAIN_FRAMES=32 P7_EPOCHS=1 \
P7_LOGDIR=outputs/local_training/p7_hf_smoke \
  bash scripts/p7/train_p7_hf_online.sh
```

For the full run, set paths and GPU list for that machine and use a fresh log
directory:

```bash
P7_LEAD_PYTHON=/path/to/miniconda3/envs/lead/bin/python \
P7_QWEN_PYTHON=/path/to/miniconda3/envs/qwendrive_official/bin/python \
P7_QWEN_MODEL=/path/to/Qwen-Drive-1.0-4B \
P7_POSTTRAIN=outputs/local_training/posttrain/model_0030.pth \
P7_LOGDIR=outputs/local_training/p7_hf_full \
P7_BATCH_SIZE=64 P7_EPOCHS=20 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/p7/train_p7_hf_online.sh
```

`P7_TRAIN_FRAMES` defaults to 0, meaning all bucket-selected CARLA frames.
`P7_INTENT_PREFETCH` defaults to 1: while the planner trains on batch `t`, a
single ordered background request generates Qwen-Drive intents for batch `t+1`.
Set it to 0 for a synchronous timing comparison. The timing log separates
`qwendrive_wait`, `qwendrive_parse`, and `next_batch_prefetch`; the first batch
still waits for Qwen-Drive before training can begin.
`P7_WORKERS` defaults to 4 per rank. The launcher starts one service per GPU,
waits for its socket, and then runs torchrun. It sets the posttrain load file,
10 route points, K=6, paired K speeds and the P7 semantic query gate. The
existing query migration copies the posttrain route queries to all K slots and
loads the shared temporal/speed queries; new semantic adapter weights start
fresh. Check the `P7 checkpoint load ...` lines in the training log.

Every 50 steps rank 0 reports the number of valid intent slots and fallback
samples. If a response fails JSON parsing or has no usable branch, only slot 0
is active and the semantic residual is disabled for that sample. Ten
consecutive all-fallback batches stop the run so a broken VQA service cannot
silently turn the experiment into LEAD-only training. Individual generation
errors are logged in `qwendrive_rankN.log`; aggregate fallback counts are in
the trainer log.

This online HF path repeats VQA generation every epoch. One-batch lookahead
overlaps generation with planner training, but the HF service still generates
each sample sequentially within a batch. No GPU end-to-end run has yet been
performed in this workspace. Before the full run, confirm the smoke batch
reaches a checkpoint and inspect the fallback rate and route/speed losses.
