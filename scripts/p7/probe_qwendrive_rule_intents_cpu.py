"""Small CPU VQA probe for rule-conditioned Qwen-Drive route intents."""

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
from probe_qwendrive_branch_set_cpu import generate_text_with_metadata

MANEUVERS = {
    "keep_lane", "change_lane_left", "change_lane_right", "turn_left", "turn_right"
}
OBLIGATIONS = {
    "stop_for_red", "stop_at_sign", "yield_pedestrian", "yield_vehicle",
    "follow_lead", "slow_for_obstacle",
}
SIGNALS = {"red", "yellow", "green", "none", "uncertain"}
TRISTATE = {"yes", "no", "uncertain"}


def analysis_prompt(command: str, speed: float) -> str:
    return (
        "Analyze these front, front-left and front-right CARLA images for the ego's "
        "next 10 meters. Give concise, concrete observations in this order: "
        "(1) ego lane, lane boundaries, oncoming direction, arrows, reachable "
        "adjacent lanes, and the navigation direction; (2) traffic lights and "
        "stop signs; (3) crosswalks, pedestrians, vehicles, roadblocks, and "
        "visibility; (4) what must happen before proceeding along each plausible "
        "route. A yellow centerline or oncoming lane is never a lane-change "
        "target. A curb, shoulder, parking area, or obstacle is not a lane. "
        "Stopping or yielding changes timing, not the road branch. Say uncertain "
        "where the single frame does not establish a fact. "
        f"Navigation: {command}. Ego speed: {speed:.1f} m/s."
    )


def final_prompt(command: str, speed: float, analysis: str) -> str:
    return (
        "Use the three images and this fallible scene analysis; verify the analysis "
        "against the images before answering: " + analysis + "\n"
        "Output ONLY one JSON object with keys scene_rules and intents. "
        "scene_rules has signal (red/yellow/green/none/uncertain), stop_sign, "
        "pedestrian_conflict, vehicle_conflict (each yes/no/uncertain). "
        "Intents is a list of at most three navigation-compatible, legally "
        "traversable routes. Each item has maneuver, obligations, confidence, "
        "evidence. maneuver is keep_lane, change_lane_left, change_lane_right, "
        "turn_left, or turn_right. A lane change moves one lane sideways but "
        "keeps the same overall direction; a turn takes a junction branch. "
        "obligations is a list from stop_for_red, stop_at_sign, "
        "yield_pedestrian, yield_vehicle, follow_lead, slow_for_obstacle. "
        "Use an empty list only when no such obligation is visible. "
        "A red light or stop sign does not erase the route: retain the route "
        "and add its mandatory stop obligation. Never offer a yellow-centerline "
        "crossing, oncoming lane, turn-only lane used for straight travel, "
        "blocked lane, road edge, or absent adjacent lane. If legality is "
        "uncertain, omit that optional route. confidence is a number 0 to 1; "
        "evidence is one short visual phrase. No duplicate maneuvers. "
        "Example: {\"scene_rules\":{\"signal\":\"red\",\"stop_sign\":\"no\","
        "\"pedestrian_conflict\":\"no\",\"vehicle_conflict\":\"uncertain\"},"
        "\"intents\":[{\"maneuver\":\"keep_lane\",\"obligations\":[\"stop_for_red\"],"
        "\"confidence\":0.8,\"evidence\":\"red signal ahead\"}]}. "
        "The example is illustrative, not an instruction to always output one route "
        "or a red signal. No markdown or explanation outside JSON. "
        f"Navigation: {command}. Ego speed: {speed:.1f} m/s."
    )


def parse_response(raw: str):
    candidate = raw.strip()
    warning = None
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
        warning = "markdown_fence"
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"json_error: {exc.msg}", warning
    if not isinstance(value, dict) or set(value) != {"scene_rules", "intents"}:
        return None, "schema_error: top-level fields", warning
    rules, intents = value["scene_rules"], value["intents"]
    if not isinstance(rules, dict) or set(rules) != {
        "signal", "stop_sign", "pedestrian_conflict", "vehicle_conflict"
    }:
        return None, "schema_error: scene rules", warning
    if (not isinstance(rules["signal"], str) or rules["signal"] not in SIGNALS
            or any(not isinstance(rules[key], str) or rules[key] not in TRISTATE
        for key in ("stop_sign", "pedestrian_conflict", "vehicle_conflict")
            )):
        return None, "schema_error: scene values", warning
    if not isinstance(intents, list) or len(intents) > 3:
        return None, "schema_error: intents list", warning
    seen = set()
    for intent in intents:
        if not isinstance(intent, dict) or set(intent) != {
            "maneuver", "obligations", "confidence", "evidence"
        }:
            return None, "schema_error: intent fields", warning
        maneuver = intent["maneuver"]
        if not isinstance(maneuver, str) or maneuver not in MANEUVERS or maneuver in seen:
            return None, "schema_error: maneuver", warning
        seen.add(maneuver)
        obligations = intent["obligations"]
        if (not isinstance(obligations, list)
                or any(not isinstance(item, str) or item not in OBLIGATIONS
                       for item in obligations)
                or len(set(obligations)) != len(obligations)):
            return None, "schema_error: obligations", warning
        confidence = intent["confidence"]
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1):
            return None, "schema_error: confidence", warning
        if not isinstance(intent["evidence"], str):
            return None, "schema_error: evidence", warning
    return value, "ok", warning


def rule_consistency(parsed: dict | None) -> list[str]:
    if parsed is None:
        return []
    rules = parsed["scene_rules"]
    issues = []
    for index, intent in enumerate(parsed["intents"]):
        obligations = set(intent["obligations"])
        if rules["signal"] == "red" and "stop_for_red" not in obligations:
            issues.append(f"intent[{index}] misses stop_for_red")
        if rules["stop_sign"] == "yes" and "stop_at_sign" not in obligations:
            issues.append(f"intent[{index}] misses stop_at_sign")
        if rules["pedestrian_conflict"] == "yes" and "yield_pedestrian" not in obligations:
            issues.append(f"intent[{index}] misses yield_pedestrian")
    return issues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--analysis-tokens", type=int, default=512)
    ap.add_argument("--answer-tokens", type=int, default=384)
    args = ap.parse_args()

    import torch
    import transformers
    from PIL import Image
    from qwen_drive import QwenDriveForPlanning

    torch.set_num_threads(16)
    torch.set_num_interop_threads(2)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    entries = [json.loads(line) for line in args.manifest.open() if line.strip()]
    for entry in entries:
        image = Path(entry["image"])
        metadata = image.parent.parent / "metas" / (image.stem + ".pkl")
        if not image.is_file() or not metadata.is_file():
            raise FileNotFoundError(f"missing image or metadata for {image}")

    start = time.monotonic()
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cpu").eval()
    print(f"loaded model in {time.monotonic() - start:.1f}s", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.output.open("w") as stream:
        for index, entry in enumerate(entries, 1):
            image_path = Path(entry["image"]).resolve()
            meta_path = image_path.parent.parent / "metas" / (image_path.stem + ".pkl")
            with lzma.open(meta_path, "rb") as meta_stream:
                meta = pickle.load(meta_stream)
            command = COMMANDS.get(int(meta["next_commands"][0]), "unknown")
            speed = float(meta["speed"])
            with Image.open(image_path) as strip:
                strip = strip.convert("RGB")
                width = strip.width // 3
                images = [strip.crop((width, 0, 2 * width, strip.height)),
                          strip.crop((0, 0, width, strip.height)),
                          strip.crop((2 * width, 0, 3 * width, strip.height))]
            started = time.monotonic()
            try:
                analysis, analysis_meta = generate_text_with_metadata(
                    model, images, analysis_prompt(command, speed), args.analysis_tokens
                )
                answer, answer_meta = generate_text_with_metadata(
                    model, images, final_prompt(command, speed, analysis), args.answer_tokens
                )
                parsed, status, warning = parse_response(answer)
                error = None
            except Exception as exc:
                analysis = answer = ""
                analysis_meta = answer_meta = None
                parsed, warning = None, None
                status = "inference_error"
                error = f"{type(exc).__name__}: {exc}"
            issues = rule_consistency(parsed)
            row = {
                "index": index, "id": entry["id"], "image": str(image_path),
                "review_focus": entry.get("review_focus"),
                "human_review": entry.get("human_review"),
                "command": command, "speed_mps": speed,
                "analysis": analysis, "analysis_raw": analysis_meta["raw"] if analysis_meta else None,
                "analysis_tokens": analysis_meta["tokens"] if analysis_meta else None,
                "analysis_hit_limit": analysis_meta["hit_limit"] if analysis_meta else None,
                "raw_text": answer, "answer_tokens": answer_meta["tokens"] if answer_meta else None,
                "answer_hit_limit": answer_meta["hit_limit"] if answer_meta else None,
                "parsed": parsed, "status": status, "format_warning": warning,
                "consistency_issues": issues, "error": error,
                "planner_usable": False,
                "seconds": round(time.monotonic() - started, 3),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            rows.append(row)
            count = len(parsed["intents"]) if parsed else 0
            print(f"{index}/{len(entries)} {entry['id']}: {status}, "
                  f"intents={count}, issues={len(issues)}, {row['seconds']:.1f}s", flush=True)

    summary = {
        "samples": len(rows), "schema_ok": sum(row["status"] == "ok" for row in rows),
        "multi_intent": sum(len(row["parsed"]["intents"]) >= 2 for row in rows if row["parsed"]),
        "rule_consistency_issues": sum(len(row["consistency_issues"]) for row in rows),
        "analysis_hit_limit": sum(row["analysis_hit_limit"] is True for row in rows),
        "answer_hit_limit": sum(row["answer_hit_limit"] is True for row in rows),
        "median_seconds": statistics.median(row["seconds"] for row in rows),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "manifest": str(args.manifest), "model": str(args.model),
    }
    args.output.with_name("summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Rule-conditioned intent review", "",
             "VLM intents are unverified. `planner_usable=false` until an independent rule/geometry gate validates them.",
             "", "| # | Scene | Image | Navigation | Intents and obligations | Status |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        intents = row["parsed"]["intents"] if row["parsed"] else []
        text = "; ".join(f"{x['maneuver']} ({', '.join(x['obligations']) or 'none'})" for x in intents) or "—"
        lines.append(f"| {row['index']} | {row['id']} | [image](<{row['image']}>) | "
                     f"{row['command']} | {text} | {row['status']} |")
    for row in rows:
        lines += ["", f"## {row['index']}. {row['id']}", "",
                  f"Focus: {row['review_focus'] or '—'}", "",
                  f"Human note: {(row.get('human_review') or {}).get('reason', 'Pending')}", "",
                  f"Analysis hit token cap: {row['analysis_hit_limit']}", "",
                  f"Analysis: {row['analysis'] or '—'}", "",
                  f"Rule consistency issues: {', '.join(row['consistency_issues']) or 'none'}", ""]
    args.output.with_name("review.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
