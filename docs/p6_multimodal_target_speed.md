# Paired route / target-speed candidates

The optional `multimodal_target_speed=true` configuration changes the joint
planner from K routes sharing one target speed to K paired route/speed candidates.
It requires `multimodal_planner=true`, spatial route and target-speed prediction,
`joint_scorer_replaces_conf=true`, and `route_select_by_conf=false`.

For each route arm, mean-pool its decoded route queries (already conditioned on
the arm anchor and planner scene). Add the decoded global speed query as shared
ego/dynamics context, then apply the shared speed MLP to each resulting arm token.
Outputs are `[B,K,C]` speed logits and `[B,K]` decoded scalar speeds. This does not
replicate speed-head parameters or add transformer queries. The one temporal
waypoint sequence and its auxiliary L1 loss remain unchanged.

Route WTA and speed supervision use the **same** closest valid route to the expert.
Only that arm's speed distribution receives the existing brake-aware two-hot
cross-entropy target. Non-winners do not receive fabricated expert speed labels;
their speed quality and mode coverage must therefore be audited after training.
The shared head/route features still receive gradients from winners across data.

The scorer constructs each candidate using its own target speed and the existing
speed-ramp rule (eight steps/2 seconds for long routes, or four steps/1 second
for the 10 m route). Same-pass labels are computed on these exact
candidates. Candidate and scene inputs remain detached: scorer gradients do not
update the planner. At inference, the scorer's selected index gathers BOTH the
route and speed distribution/scalar. Single-route output slots are slot-0
placeholders during training; speed loss and training speed metrics explicitly
gather the WTA arm. Full candidate speeds are also exposed in `Prediction`.

The default remains `false` for old runs/checkpoints. To train the new structure,
add `P6_JOINT_MULTIMODAL_SPEED=true` to the existing wrapper command and use a new
log directory, for example `outputs/local_training/p6_joint_online_k_speed`.
For the 10 m experiment, set `P6_JOINT_ROUTE_POINTS=10`; the wrapper defaults
to `P6_JOINT_SCORER_STEPS=4`. Initialize from the posttrain checkpoint
(`P6_JOINT_PRETRAIN`, default `posttrain/model_0030.pth`) and the frozen intent
decoder. This loads model weights but starts a new optimizer and scheduler.
Do not use `P6_JOINT_RESUME` to convert a
single-speed training run into a paired-speed run; the wrapper rejects that
configuration change. Resume within the same structure is supported.

This resolves forced speed sharing, not the remaining discrepancy between the
heuristic rollout and actual PID execution. It does not produce K temporal
waypoint sequences or complete learned velocity profiles, and does not by itself
guarantee a closed-loop improvement. Model ensembling/control wrappers must use
the selected paired outputs, not independently select a speed arm.
