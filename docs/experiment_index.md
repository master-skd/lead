# Experiment code and documents

The repository contains successive research experiments. Use this index to find
the relevant entry point without treating an older diagnostic as the current
training pipeline.

| Track | Main entry point | Model code | Reference |
| --- | --- | --- | --- |
| Original LEAD | `scripts/posttrain_ddp.sh` | `lead/tfv6/` | `docs/source/carla_training.md` |
| P5 corridor-conditioned multi-route planner | `scripts/p4/train_p5_stepB2_corridor_ddp.sh` | `lead/tfv6/planning_decoder.py`, `lead/tfv6/anchor_extraction.py` | `docs/p6_candidate_diagnostic_eval.md` |
| P4/P5 DrivoR-style trajectory scorer | `scripts/p4/train_p5_stepB3a_v2_scene_scorer.sh` | `lead/tfv6/trajectory_scene_scorer.py` | `docs/p4_b3a_v2_scene_scorer.md` |
| P6 online intent + joint scorer | `scripts/p4/train_p6_joint_online.sh` | `lead/tfv6/joint_trajectory_scorer.py`, `lead/tfv6/tfv6.py` | `docs/p6_joint_model_architecture.md`, `docs/p6_multimodal_target_speed.md` |
| P7 Qwen-Drive semantic intent | `scripts/p7/train_p7_hf_online.sh` | `lead/tfv6/semantic_intent_encoder.py`, `lead/tfv6/semantic_intent_slots.py` | `docs/p7_hf_training.md`, `P7_QwenDrive_AR_Intent_TFv6_Design.md` |

The P5 corridor and P4/P5 scorer implementations and their reproducibility
scripts are retained. The P6 launcher keeps the optional HF, vLLM service and
vLLM pooling backends; the service entry is
`scripts/p4/start_p6_vllm_hidden_service.sh`. The removed vLLM pilot benchmark
and one-off hidden-state parity scripts were experiment checks, not dependencies
of the launcher. P6 candidate diagnostics still use
`scripts/p4/check_p6_intent_capture.py` and
`scripts/p4/summarize_p6_candidate_trace.py`.

The P7 HF output ingestion and empty-intent training fallback are implemented.
Full GPU training and closed-loop evaluation are still pending.
Historical P4/P5/P6 reports under `docs/` record their own experimental
conditions; they should not be read as P7 training instructions.
