"""Compare Qwen-Drive route candidates and traffic state on reviewed frames."""

from __future__ import annotations

import argparse
import json
import lzma
import pickle
import re
import statistics
import time
import warnings
from pathlib import Path

from probe_qwendrive_ar_cpu import COMMANDS

PROMPT_VERSION = "p7_route_traffic_cpu_probe_v1"
PROMPT_VERSION_V2 = "p7_route_traffic_cpu_probe_v2"
PROMPT_VERSION_V3 = "p7_route_traffic_cpu_probe_v3"
PROMPT_VERSION_V4 = "p7_route_traffic_cpu_probe_v4"
ROUTES = {
    "keep_lane", "turn_left", "turn_right", "change_lane_left", "change_lane_right"
}
YES_NO_UNCERTAIN = {"yes", "no", "uncertain"}
CAUSES = {"red_light", "stop_sign", "vehicle", "pedestrian", "obstacle", "none", "uncertain"}


def prompt(command: str, speed: float, version: str = PROMPT_VERSION) -> str:
    if version == PROMPT_VERSION_V4:
        return (
            prompt(command, speed, PROMPT_VERSION_V3)
            .replace(
                "Check each proposed lane change:",
                "Before any left lane change, check whether a yellow centerline "
                "separates oncoming traffic; a vehicle facing the ego confirms "
                "that lane is not an alternative. A road edge, parked-vehicle "
                "area, or shoulder is not a target lane. Identify the ego's "
                "position relative to the leftmost and rightmost travel lanes "
                "before proposing either side. Check each proposed lane change:",
            )
            .replace(
                "cause is red_light, vehicle, pedestrian, obstacle, none, or uncertain.",
                "cause is red_light, stop_sign, vehicle, pedestrian, obstacle, "
                "none, or uncertain. For a stop sign use cause=stop_sign. "
                "For no yielding use yield_required=no, never none.",
            )
        )
    if version == PROMPT_VERSION_V3:
        return (
            "The three synchronized CARLA images are front, front-left, and front-right. "
            "List distinct feasible ego paths for the next roughly 10 meters that "
            "continue in the navigation direction. keep_lane stays in the ego's "
            "current lane. change_lane_left or change_lane_right moves exactly one "
            "lane sideways into an existing adjacent lane, then continues in the "
            "same overall direction. These are straight-through alternatives, not "
            "left or right turns. turn_left and turn_right mean taking the respective "
            "branch at a junction; a turn-only lane cannot count as a straight-through "
            "lane change. In particular, a right-turn-only arrow rules out "
            "change_lane_right for a straight-through route, and a left-turn-only "
            "arrow rules out change_lane_left. Read pavement arrows and lane signs "
            "before adding a lane. "
            "A follow_lane command permits optional one-lane changes if the target "
            "lane continues with the route. A change_lane_left/right command expresses "
            "a preference for that lane, but do not remove other physically feasible "
            "straight-through lanes solely because of this command. A turn_left/right "
            "command requires the corresponding junction turn; omit incompatible "
            "straight or opposite-turn paths. Check each proposed lane change: the "
            "target lane must exist immediately next to the ego, the boundary must "
            "be crossable, its arrow/sign must allow the navigation direction, and "
            "there must be enough road to reach it. If the ego is already in the "
            "rightmost lane, never output change_lane_right; likewise for the leftmost "
            "lane and change_lane_left. Do not count a distant lane requiring two "
            "lane changes as one candidate. If any condition is unclear, omit that "
            "optional path. A vehicle, red light, stop, or yield affects timing, not "
            "which road branches exist; report traffic separately. Return ONLY one "
            "JSON object. For example: {\"routes\":[{\"maneuver\":\"keep_lane\","
            "\"confidence\":0.9},{\"maneuver\":\"change_lane_left\","
            "\"confidence\":0.7}],\"traffic\":{\"stop_required\":\"no\","
            "\"yield_required\":\"uncertain\",\"cause\":\"vehicle\"}}. "
            "One route is valid when only one passes these checks. Use at most three "
            "distinct maneuvers from keep_lane, change_lane_left, change_lane_right, "
            "turn_left, turn_right. Confidence is 0 to 1. stop_required and "
            "yield_required are yes, no, or uncertain. cause is red_light, vehicle, "
            "pedestrian, obstacle, none, or uncertain. No explanation or markdown. "
            f"Navigation: {command}. Ego speed: {speed:.1f} m/s."
        )
    if version == PROMPT_VERSION_V2:
        return (
            "The three synchronized CARLA images are front, front-left, and front-right. "
            "List distinct, visible, navigation-compatible ego paths for approximately "
            "the next 10 meters. Check the current lane, then the immediately adjacent "
            "left and right lanes separately. keep_lane means remaining in the current "
            "lane; change_lane_left and change_lane_right mean moving into the respective "
            "adjacent same-direction lane and continuing forward. Under follow_lane "
            "navigation, include a lane change as an optional route when the adjacent "
            "lane is visibly reachable, the lane boundary permits crossing, and the "
            "change still follows the navigation. It need not be required by the "
            "navigation command or improve speed. For a turn or lane-change command, "
            "exclude any route that would miss that command. Do not infer crossing "
            "permission from the mere presence of another lane. If markings, lane "
            "direction, or reachability are unclear, omit that optional lane change. "
            "A stop or yield decision changes timing, not the route set; report traffic "
            "control separately. Never invent a route just to produce multiple options. "
            "Return ONLY a JSON object with routes and traffic. A valid two-route "
            "example is: {\"routes\":[{\"maneuver\":\"keep_lane\",\"confidence\":0.9},"
            "{\"maneuver\":\"change_lane_left\",\"confidence\":0.7}],"
            "\"traffic\":{\"stop_required\":\"no\",\"yield_required\":\"uncertain\","
            "\"cause\":\"vehicle\"}}. A one-route list is equally valid. "
            "Use at most three distinct maneuvers from keep_lane, turn_left, turn_right, "
            "change_lane_left, change_lane_right. Confidence is a number from 0 to 1. "
            "stop_required and yield_required are yes, no, or uncertain. cause is "
            "red_light, vehicle, pedestrian, obstacle, none, or uncertain. "
            "No explanation or markdown. "
            f"Navigation: {command}. Ego speed: {speed:.1f} m/s."
        )
    if version != PROMPT_VERSION:
        raise ValueError(f"unknown prompt version: {version}")
    return (
        "The three synchronized CARLA images are front, front-left, and front-right. "
        "Identify visible, navigation-compatible ego ROUTES for roughly the next "
        "10 meters, and assess traffic control separately. A route is a distinct "
        "path through the road, not a different speed or waiting decision. "
        "For follow_lane navigation, a visible adjacent same-direction lane can "
        "be an alternative if a lane change is permitted and it preserves the "
        "navigation route. For a turn or lane-change instruction, reject routes "
        "that would miss that instruction. Return one route when only one is "
        "supported; never invent lanes or force multiple routes. Stopping at a "
        "red light does not remove the route beyond the light. Return ONLY JSON: "
        "{\"routes\":[{\"maneuver\":\"keep_lane\",\"confidence\":0.8}],"
        "\"traffic\":{\"stop_required\":\"no\",\"yield_required\":\"uncertain\","
        "\"cause\":\"vehicle\"}}. The routes array has at most three items. "
        "Each maneuver must be exactly one of keep_lane, turn_left, turn_right, "
        "change_lane_left, change_lane_right and must not repeat. confidence is "
        "a number from 0 to 1. stop_required and yield_required must each be "
        "yes, no, or uncertain. cause must be red_light, vehicle, pedestrian, "
        "obstacle, none, or uncertain. Use uncertain when visibility is inadequate. "
        "No explanation or markdown. "
        f"Navigation: {command}. Ego speed: {speed:.1f} m/s."
    )


def parse_output(raw: str) -> tuple[dict | None, str, str | None]:
    candidate = raw.strip()
    format_warning = None
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
        format_warning = "markdown_fence"
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"json_error: {exc.msg}", format_warning
    if not isinstance(value, dict) or set(value) != {"routes", "traffic"}:
        return None, "schema_error: top-level keys", format_warning
    routes, traffic = value["routes"], value["traffic"]
    if not isinstance(routes, list) or len(routes) > 3 or not isinstance(traffic, dict):
        return None, "schema_error: routes or traffic", format_warning
    seen = set()
    for item in routes:
        if not isinstance(item, dict) or set(item) != {"maneuver", "confidence"}:
            return None, "schema_error: route fields", format_warning
        maneuver, confidence = item["maneuver"], item["confidence"]
        if maneuver not in ROUTES or maneuver in seen:
            return None, "schema_error: route vocabulary or duplicate", format_warning
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return None, "schema_error: confidence type", format_warning
        if not 0 <= confidence <= 1:
            return None, "schema_error: confidence range", format_warning
        seen.add(maneuver)
    if set(traffic) != {"stop_required", "yield_required", "cause"}:
        return None, "schema_error: traffic fields", format_warning
    if (traffic["stop_required"] not in YES_NO_UNCERTAIN
            or traffic["yield_required"] not in YES_NO_UNCERTAIN
            or traffic["cause"] not in CAUSES):
        return None, "schema_error: traffic vocabulary", format_warning
    return value, "ok", format_warning


def split_thinking(raw: str) -> tuple[str | None, str | None]:
    """Return the reasoning and final answer, preserving incomplete traces."""
    if "</think>" not in raw:
        return raw.removeprefix("<think>").strip(), None
    reasoning, answer = raw.split("</think>", 1)
    answer = answer.split("<|im_end|>", 1)[0]
    answer = answer.split("<|endoftext|>", 1)[0].strip()
    return reasoning.removeprefix("<think>").strip(), answer


def generate_text_with_metadata(model, images, question: str, max_new_tokens: int):
    """Mirror official VQA decoding while retaining token count and stop state."""
    import torch
    from qwen_drive.modeling_qwen_drive import VQA_DECODE_DEFAULTS

    processor = model.processor
    inputs = processor.encode_vqa(images, question, device=model.device)
    params = dict(VQA_DECODE_DEFAULTS)
    params["max_new_tokens"] = max_new_tokens
    seed = params.pop("seed", None)
    if seed is not None:
        torch.manual_seed(seed)
    params.pop("presence_penalty", None)
    with torch.no_grad():
        generated = model.vlm.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            mm_token_type_ids=model._modality_ids(inputs["input_ids"]),
            **params,
        )
    tokens = generated[0, inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.decode(tokens, skip_special_tokens=False)
    answer = processor.tokenizer.decode(tokens, skip_special_tokens=True)
    answer = answer.split("</think>")[-1].strip()
    eos = model.vlm.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    eos_ids.add(processor.im_end_id)
    ended = bool(tokens.numel() and int(tokens[-1]) in eos_ids)
    return answer, {
        "raw": raw,
        "tokens": int(tokens.numel()),
        "ended": ended,
        "hit_limit": bool(tokens.numel() >= max_new_tokens and not ended),
    }


def generate_with_reasoning_pass(
    model, images, question: str, command: str, speed: float,
    reasoning_tokens: int, answer_tokens: int, analysis_version: str,
):
    """Use an explicit visual-analysis pass before the final structured answer."""
    analysis_prompt = (
        "The images are front, front-left, and front-right CARLA views. "
        "Analyze the next roughly 10 meters in 2 to 4 concrete sentences. "
        "Locate the ego's current lane and its immediate left and right neighbors. "
        "Describe visible lane arrows, whether each neighboring lane continues "
        "in the navigation direction, and whether the ego is at a road edge. "
        "Separate a lane change from a junction turn. Note any traffic signal "
        "or actor that requires stopping, without calling stopping a route. "
        "This is a visual analysis, not the final JSON answer. "
        f"Navigation: {command}. Ego speed: {speed:.1f} m/s."
    )
    if analysis_version == "v2":
        analysis_prompt += (
            " Before deciding either lane change, identify yellow or double-yellow "
            "centerlines and the facing direction of nearby cars; an oncoming lane "
            "is never a target. A curb, shoulder, parked-car area, or barrier is "
            "not a drivable neighboring lane. Locate the ego in the camera-center "
            "view and check whether it is already in the leftmost or rightmost "
            "travel lane. End with explicit left-feasible and right-feasible "
            "judgments supported by image evidence."
        )
    analysis, analysis_meta = generate_text_with_metadata(
        model, images, analysis_prompt, reasoning_tokens
    )
    if not analysis.strip():
        return analysis, None, analysis_meta, None
    final_question = (
        "First-pass scene analysis (check it against the images; it can be wrong): "
        + analysis.strip() + "\nFinal task: " + question
        + " Stopping is never a route maneuver. Return only the final JSON."
    )
    answer, answer_meta = generate_text_with_metadata(
        model, images, final_question, answer_tokens
    )
    return analysis, answer, analysis_meta, answer_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--thinking", action="store_true",
                    help="run an explicit visual-analysis pass before final JSON")
    ap.add_argument("--thinking-max-new-tokens", type=int, default=256,
                    help="token budget for the visual-analysis pass")
    ap.add_argument("--analysis-version", choices=["v1", "v2"], default="v1")
    ap.add_argument("--prompt-version", choices=[PROMPT_VERSION, PROMPT_VERSION_V2,
                                                  PROMPT_VERSION_V3, PROMPT_VERSION_V4],
                    default=PROMPT_VERSION)
    ap.add_argument("--resume", action="store_true", help="reuse and reparse completed JSONL rows")
    args = ap.parse_args()

    import torch
    import transformers
    from PIL import Image
    from qwen_drive import QwenDriveForPlanning

    torch.set_num_threads(16)
    torch.set_num_interop_threads(2)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    entries = [json.loads(line) for line in args.annotations.open()]
    groups = {entry["group"] for entry in entries}
    if not entries or not groups <= {"single", "multi", "review"}:
        raise ValueError("annotations must use single, multi, or review groups")
    for entry in entries:
        path = Path(entry["image"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if entry["group"] == "single" and len(entry["annotation"]["route_maneuvers"]) != 1:
            raise ValueError(f"invalid route annotation: {entry['id']}")
        if entry["group"] == "multi" and len(entry["annotation"]["route_maneuvers"]) < 2:
            raise ValueError(f"multi annotation has fewer than two routes: {entry['id']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = []
    if args.resume and args.output.exists():
        previous = [json.loads(line) for line in args.output.open() if line.strip()]
        for index, row in enumerate(previous):
            if index >= len(entries) or row["id"] != entries[index]["id"]:
                raise ValueError("existing results do not match the annotation order")
            if (row.get("prompt_version") != args.prompt_version
                    or bool(row.get("thinking")) != args.thinking
                    or (args.thinking and (
                        row.get("thinking_mode") != "two_pass"
                        or row.get("analysis_version") != args.analysis_version))):
                raise ValueError("existing results use a different prompt or thinking mode")
            parsed, status, format_warning = parse_output(row["raw_text"])
            predicted = [item["maneuver"] for item in parsed["routes"]] if parsed else []
            row.update(
                parsed=parsed, status=status, format_warning=format_warning,
                predicted_route_maneuvers=predicted,
                route_set_exact=(None if entries[index]["group"] == "review" else
                                 status == "ok" and set(predicted) ==
                                 set(entries[index]["annotation"]["route_maneuvers"])),
            )
        print(f"reusing {len(previous)} completed frames", flush=True)

    start = time.monotonic()
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cpu").eval()
    load_seconds = time.monotonic() - start
    print(f"loaded model in {load_seconds:.1f}s", flush=True)

    rows = list(previous)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in previous:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        for index, entry in enumerate(entries[len(previous):], len(previous) + 1):
            image_path = Path(entry["image"]).resolve()
            meta_path = image_path.parent.parent / "metas" / (image_path.stem + ".pkl")
            with lzma.open(meta_path, "rb") as meta_stream:
                meta = pickle.load(meta_stream)
            command_id = int(meta["next_commands"][0])
            command = COMMANDS.get(command_id, "unknown")
            speed = float(meta["speed"])
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                if image.width % 3:
                    raise ValueError(f"three-camera strip width not divisible by 3: {image_path}")
                width = image.width // 3
                left = image.crop((0, 0, width, image.height))
                front = image.crop((width, 0, 2 * width, image.height))
                right = image.crop((2 * width, 0, 3 * width, image.height))
            started = time.monotonic()
            try:
                question = prompt(command, speed, args.prompt_version)
                if args.thinking:
                    reasoning, answer, analysis_meta, answer_meta = generate_with_reasoning_pass(
                        model, [front, left, right], question, command, speed,
                        args.thinking_max_new_tokens, args.max_new_tokens,
                        args.analysis_version,
                    )
                    raw = answer or ""
                    raw_generation = answer_meta["raw"] if answer_meta else None
                    generated_tokens = answer_meta["tokens"] if answer_meta else None
                    if answer is None:
                        parsed, status, format_warning = None, "thinking_incomplete", None
                    else:
                        parsed, status, format_warning = parse_output(raw)
                else:
                    response = model.generate_text(
                        [front, left, right], question,
                        max_new_tokens=args.max_new_tokens,
                    )
                    raw = response.text or ""
                    raw_generation, reasoning, generated_tokens = None, None, None
                    analysis_meta, answer_meta = None, None
                    parsed, status, format_warning = parse_output(raw)
                error = None
            except Exception as exc:
                raw, parsed = "", None
                status, error = "inference_error", f"{type(exc).__name__}: {exc}"
                format_warning = None
                raw_generation, reasoning, generated_tokens = None, None, None
                analysis_meta, answer_meta = None, None
            predicted = [item["maneuver"] for item in parsed["routes"]] if parsed else []
            expected = entry["annotation"]["route_maneuvers"]
            row = {
                "index": index, "id": entry["id"], "group": entry["group"],
                "image": str(image_path), "command_id": command_id,
                "command": command, "speed_mps": speed,
                "review_focus": entry.get("review_focus"),
                "human_review": entry.get("human_review"),
                "prompt_version": args.prompt_version,
                "thinking": args.thinking,
                "thinking_mode": "two_pass" if args.thinking else None,
                "analysis_version": args.analysis_version if args.thinking else None,
                "annotation": entry["annotation"],
                "metadata_hints": {
                    "lane_change_str": meta.get("lane_change_str"),
                    "traffic_light_state": meta.get("traffic_light_state"),
                },
                "raw_text": raw, "parsed": parsed, "status": status,
                "raw_generation": raw_generation, "reasoning_text": reasoning,
                "generated_tokens": generated_tokens,
                "analysis_generation": analysis_meta["raw"] if analysis_meta else None,
                "analysis_generated_tokens": analysis_meta["tokens"] if analysis_meta else None,
                "analysis_hit_limit": analysis_meta["hit_limit"] if analysis_meta else None,
                "answer_hit_limit": answer_meta["hit_limit"] if answer_meta else None,
                "format_warning": format_warning,
                "error": error, "predicted_route_maneuvers": predicted,
                "route_set_exact": (None if entry["group"] == "review" else
                                    status == "ok" and set(predicted) == set(expected)),
                "seconds": round(time.monotonic() - started, 3),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            rows.append(row)
            print(f"{index}/{len(entries)} {entry['id']}: {status}, "
                  f"expected={expected}, predicted={predicted}, "
                  f"{row['seconds']:.1f}s", flush=True)

    summary = {
        "prompt_version": args.prompt_version,
        "thinking": args.thinking,
        "thinking_mode": "two_pass" if args.thinking else None,
        "analysis_version": args.analysis_version if args.thinking else None,
        "model": str(args.model), "annotations": str(args.annotations),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "model_load_seconds": round(load_seconds, 3),
        "samples": len(rows),
        "review_samples": sum(row["group"] == "review" for row in rows),
        "labeled_samples": sum(row["group"] != "review" for row in rows),
        "schema_ok": sum(row["status"] == "ok" for row in rows),
        "strict_json": sum(row["status"] == "ok" and not row["format_warning"] for row in rows),
        "fenced_json": sum(row["format_warning"] == "markdown_fence" for row in rows),
        "thinking_complete": sum(bool(row.get("reasoning_text") and row.get("raw_text"))
                                 for row in rows),
        "analysis_hit_limit": sum(row.get("analysis_hit_limit") is True for row in rows),
        "answer_hit_limit": sum(row.get("answer_hit_limit") is True for row in rows),
        "route_set_exact": sum(row["route_set_exact"] is True for row in rows),
        "single_route_set_exact": sum(row["route_set_exact"] is True for row in rows if row["group"] == "single"),
        "multi_route_set_exact": sum(row["route_set_exact"] is True for row in rows if row["group"] == "multi"),
        "multi_predicted_at_least_two": sum(len(row["predicted_route_maneuvers"]) >= 2
                                           for row in rows if row["group"] == "multi"),
        "predicted_at_least_two": sum(len(row["predicted_route_maneuvers"]) >= 2
                                      for row in rows),
        "known_stop_correct": sum(
            row["parsed"] is not None
            and row["parsed"]["traffic"]["stop_required"] == row["annotation"]["stop_required"]
            for row in rows if row["annotation"]["stop_required"] in {"yes", "no"}
        ),
        "known_stop_count": sum(row["annotation"]["stop_required"] in {"yes", "no"} for row in rows),
        "latency_median_seconds": statistics.median(row["seconds"] for row in rows),
    }
    summary_path = args.output.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if any(row["group"] == "review" for row in rows):
        review_lines = [
            "# Qwen-Drive second-set route review", "",
            "These frames are unlabelled. Candidate count and exact route sets are for human review, not accuracy scoring.",
            "", "| # | Scene | Image | Navigation | Predicted routes | Traffic | Status |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            routes = ", ".join(row["predicted_route_maneuvers"]) or "—"
            traffic = row["parsed"]["traffic"] if row["parsed"] else None
            traffic_text = (f"stop={traffic['stop_required']}, yield={traffic['yield_required']}, "
                            f"cause={traffic['cause']}") if traffic else "—"
            review_lines.append(
                f"| {row['index']} | {row['id']} | [image](<{row['image']}>) | "
                f"{row['command']} | {routes} | {traffic_text} | {row['status']} |"
            )
        for row in rows:
            review_lines += ["", f"## {row['index']}. {row['id']}", "",
                             f"Focus: {row.get('review_focus') or '—'}", "",
                             f"Human review: {(row.get('human_review') or {}).get('reason', 'Pending')}", "",
                             f"Analysis token limit hit: {row.get('analysis_hit_limit')}", "",
                             f"Analysis: {row.get('reasoning_text') or '—'}", ""]
        args.output.with_name("review.md").write_text("\n".join(review_lines) + "\n")
    print(f"wrote {args.output} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
