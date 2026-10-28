# Qwen-Drive route/traffic probe (8 frames)

This probe checks whether a frozen Qwen-Drive model distinguishes one versus multiple navigation-compatible **route geometries** while reporting stopping and yielding separately. It is an interface check, not an accuracy benchmark.

## Selection and annotations

The fixed set is [`data/p7/branch_probe_8.jsonl`](../data/p7/branch_probe_8.jsonl): four provisional single-route frames and four provisional multi-route frames. Selection used the current `metas/*.pkl` `next_commands[0]`, lane-change permission, and camera image. The manifest's `command` field was not used because it differed from `next_commands[0]` on some candidate frames. Each annotation contains:

- `route_maneuvers`: geometrically distinct, navigation-compatible candidate maneuvers; speed, waiting, and yielding do not create extra routes.
- `stop_required`, `yield_required`, `traffic_cause`: separate traffic annotations, with `uncertain` where the single frame does not establish the answer.
- `evidence` and `status`: why the case was selected and that the labels are provisional visual/metadata review, not simulator ground truth.

The model receives only the three cameras, current navigation command, and ego speed. It never receives the annotation or lane-change metadata. Output JSONL includes absolute image paths, the raw model response, parsed route/traffic fields, and the provisional comparison.

## Current limits

The multi-route group uses adjacent-lane choices under `follow_lane`. It tests one route versus multiple route maneuvers without treating a red-light stop as a second route. The fixed maneuver vocabulary cannot represent two different geometries with the **same** maneuver class (for example, two left-turn lanes). The eight annotations should be independently reviewed against the map and a longer temporal window before being used as training labels or a final accuracy claim.

The runner is [`scripts/p7/probe_qwendrive_branch_set_cpu.py`](../scripts/p7/probe_qwendrive_branch_set_cpu.py). It writes `results.jsonl`, `summary.json`, and a console log under `outputs/local_training/p7_qwendrive_branch_probe_8/`.

## Prompt and annotation review

The first prompt gave only a one-route JSON example. It produced no multi-route
prediction on four provisional multi-route frames. The second prompt explicitly
checked adjacent lanes and gave a two-route example; it produced multiple routes
on four frames, but sometimes added an invalid lane change.

The reviewed comparison manifest is
[`data/p7/branch_probe_8_reviewed_v3.jsonl`](../data/p7/branch_probe_8_reviewed_v3.jsonl).
The user identified a turn-only pavement arrow in frame 1, three forward choices
in frame 2, and the ego in the rightmost lane in frame 6. Frames 7 and 8 were
also reviewed as plausible. These judgments supersede the earlier provisional
labels for this small probe. Frame 2 is now in the multi-route group; the
`change_lane_left` navigation command is treated as a preferred lane move, while
`turn_left/right` commands constrain the junction direction. A lane change means
moving exactly one lane sideways while continuing in the same overall direction;
it cannot use a turn-only lane or an absent lane beyond the road edge.

Prompt v3 adds these rules. It still sees only camera images, navigation command,
and ego speed. The map's `lane_change_str` is a review hint in the output, not
model input. Visual review of one frame does not establish legal crossing or
future lane continuity conclusively, so this remains a diagnostic comparison.

The v3 direct run returned a schema error on the red-light frame because the
model put `stop` in the route list instead of keeping the route and setting
`traffic.stop_required=yes`. An initial attempt to prefill `<think>` was stopped
after two frames: both returned an empty `<think></think>` block. Its parser
also left `<|endoftext|>` after otherwise valid JSON, producing two artificial
`json_error: Extra data` statuses. The raw records are retained for diagnosis.

The replacement `--thinking` comparison is an explicit two-pass VQA test on the
same reviewed frames and v3 route definitions. Pass 1 asks for a short visual
analysis of lane geometry, arrows, road edges, and traffic. Pass 2 receives
that analysis as a fallible note and returns the final JSON. `reasoning_text`
and `raw_text` store these two outputs separately. This is prompted reasoning
within the VQA interface, not the official `REASONING_PLANNING` trajectory mode.

## Independent second set

[`data/p7/branch_probe_8_second_set_unreviewed.jsonl`](../data/p7/branch_probe_8_second_set_unreviewed.jsonl)
contains eight new frames from accident, side-hazard, pedestrian, red-light,
highway-exit, multilane, and right-turn scenes. All image and metadata files
were checked, and none repeats the original eight or the earlier ten-frame VQA
probe. The entries use `group=review` and intentionally carry no route labels.
The runner records model predictions without computing exact-set accuracy for
these frames and writes a `review.md` table with image links and traffic fields.

The first second-set run exposed three route errors noted by the user: the
accident frame's left lane carries oncoming traffic; the pedestrian frame has
a yellow centerline to the left; and the rainy multilane frame puts the ego in
the leftmost travel lane. The model proposed `change_lane_left` in all three.
The two traffic-schema errors were `yield_required=none` (allowed values are
yes/no/uncertain) and `cause=stop` at a stop sign (the schema lacked a stop-sign
cause). The raw eight-frame results remain unchanged. Prompt v4 adds explicit
oncoming-lane, centerline, road-edge, and obstacle checks; its cause vocabulary
adds `stop_sign` and instructs the model to write `yield_required=no` rather
than `none`. The five affected frames are in
[`data/p7/branch_probe_5_review_corrections.jsonl`](../data/p7/branch_probe_5_review_corrections.jsonl).
The targeted rerun logs each analysis generation's token count and whether it
reached the token cap, because the original two-pass run did not record this.
