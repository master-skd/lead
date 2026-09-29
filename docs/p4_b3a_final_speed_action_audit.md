# Final fixed-Path scalar-speed feasibility check

This is the last offline check before deciding whether to continue the
Path/Velocity-separated policy. It does not train a scorer or alter closed-loop
control. The confidence-selected Path remains fixed. Candidate commands are
the raw target speed, -0.5/-1/-2 m/s, stop, and +1 m/s; the rescue policy only
permits moving slowdowns. The existing control-aware oracle supplies GT-actor
collision times, while LEAD's longitudinal PID decision plus approximate CARLA
acceleration supplies traveled distance at 1.0, 1.5, and 2.0 seconds. An action
cannot count as a rescue if it outruns the route geometry cached in the old
feature shard. This avoids another model forward or Path-sidecar extraction.
The cached samples may themselves contain extrapolation from the original
full Path; this screen only prevents *additional* extrapolation beyond the
samples and cannot certify the true route endpoint.

Before running, the coverage screen was set to require at least 50
geometry-qualified raw-collision frames and a moving rescue on at least 20%
of them without exceeding the progress-loss budget. Passing it is only a
necessary condition: GT actor futures, approximate vehicle dynamics, and a
fixed Path make this an upper bound, not a deployable policy or closed-loop
performance prediction.

```bash
/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead/bin/python \
  scripts/p4/audit_b3a_short_horizon_actions.py
/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead/bin/python \
  scripts/p4/audit_b3a_short_horizon_actions.py \
  --max-progress-loss-m 2.0 \
  --output outputs/local_training/p5_stepB3a_control_aware_oracle/short_horizon_5000_loss2.json
```

The 5,000 route-disjoint held-out frames produced:

| Horizon | Qualified raw collisions | Moving rescues, <=1 m loss | Moving rescues, <=2 m loss |
| --- | ---: | ---: | ---: |
| 1.0 s | 44 | 1 (2.3%) | 2 (4.5%) |
| 1.5 s | 67 | 12 (17.9%) | 16 (23.9%) |
| 2.0 s | 90 | 2 (2.2%) | 16 (17.8%) |

All 5,000 frames have cached geometry at 1.0 s, 4,781 at 1.5 s, and 3,538
at 2.0 s. The 2.0 s, <=2 m result exactly reproduces the original oracle's
16/90. The 1.5 s, <=2 m setting narrowly passes the predeclared coverage
screen, but the Wilson 95% interval is 15.3-35.3%, only 16/5,000 total
frames are rescued, and this assumes an oracle trigger with GT actor motion.
The 1.0 s screen lacks the predeclared 50 unsafe frames and is inconclusive;
the 2.0 s screen fails under either progress budget.

The existing predicted-actor gate audit found only 26 rescues among 83
GT-rescuable collision frames on its own 5,000-frame sample, with 47 missed
because predicted risk did not flag the raw trajectory. Those samples are not
frame-aligned with this audit, so their counts must not be multiplied; they
instead show that deployable risk observability remains a separate bottleneck.

**Decision:** Do not start another velocity-only scorer training run or full
feature re-extraction based on this evidence. The narrow 1.5 s oracle window
does not offset weak risk observability and the closed-loop regressions from
frequent velocity switching. Preserve Visual Intent's Path prior, and evaluate
the next idea at the complete executable Route/trajectory level, including
controller response, safety, progress, and comfort within one candidate score.
