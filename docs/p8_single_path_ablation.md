# P8 single-route ablation with the original LEAD planner heads

The reported K-mode P8 result is DS 90.509 and 166/220 successful routes. This
alone does not identify which component reduced closed-loop performance. The
experiment combines Qwen conditioning, latent modes, WTA, confidence selection,
paired route-conditioned speed heads, and fine-tuning on a tenth of the frames.

This ablation retains the existing frozen hidden features and the same posttrain
initialization/data/epochs, and restores the original single-route planner:

- One route, one temporal waypoint sequence, one global target-speed distribution.
- Original LEAD route/waypoint/speed query layout. The 10 m configuration has
  10 route + 8 temporal + 1 speed queries, with their posttrain weights copied directly.
- The current HiddenIntentEncoder now emits one continuous condition, with the
  same adapter/cross-attention/projector/small gate. It conditions route queries.
- The speed head consumes the original global speed query. It does not add a route
  feature to that query as the K-speed architecture does.
- Expert route L1/FDE, temporal waypoint L1, and the original two-hot speed loss.
  There is no route WTA assignment, confidence head/loss or candidate selection.
- TFv6 parameters continue to fine-tune. "Original planner" describes its query
  layout, output heads and supervision, not freezing the planner weights.

One mode query is still used inside the external continuous condition encoder;
there are no replicated planner route queries. The K-mode P8 implementation remains
available under the original configuration. Single-route checkpoints use
`hidden_intent_single_path=true`, `multimodal_planner=false` and
`multimodal_target_speed=false`.

## Train on the prepared nlp_hdd copy

Reuse the completed hidden cache. No feature extraction is needed:

```bash
cd /mmu_nlp_hdd/liuzihan08/lead
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
P8_POSTTRAIN=outputs/local_training/posttrain/model_0030.pth \
P8_HIDDEN_CACHE=data/p8/hidden_cache \
P8_BATCH_SIZE=64 P8_WORKERS=8 P8_EPOCHS=20 \
bash scripts/p8/run_p8_nlp.sh train-single
```

Default output: `outputs/local_training/p8_hidden_single_cached_tenth`.
Start from the original posttrain checkpoint, rather than the K-mode trained model.
Use the same frame cache for a direct comparison. LR remains the existing P8 default
unless explicitly overridden; changes to LR/backbone freezing are separate experiments.
On other machines, set `P8_LEAD_PYTHON` and run
`bash scripts/p8/train_p8_single_cached.sh` with equivalent paths.

## Closed loop after training

The existing P8 launcher reads the checkpoint's single-route flag and uses the same
fresh per-frame Qwen forward. It runs the original single route/speed directly,
with no confidence selection:

```bash
B2D_ROUTE_IDS=3904 bash scripts/p8/eval_p8_b2d.sh \
  outputs/local_training/p8_hidden_single_cached_tenth/model_0019.pth \
  "0" p8_single_smoke
```

Use the environment/CARLA settings in [the P8 evaluation guide](p8_b2d_evaluation.md).
Per-frame traces mark `selection=single_route`. Drop `B2D_ROUTE_IDS` and use a new
tag for the full 220 routes after the smoke test. Improved scores remain an
experimental question.
