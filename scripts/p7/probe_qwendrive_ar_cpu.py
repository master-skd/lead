"""Small CPU probe for Qwen-Drive's VQA-only semantic-intent interface."""

from __future__ import annotations

import argparse
import json
import lzma
import pickle
import statistics
import sys
import time
import warnings
from pathlib import Path

SCENARIOS = (
    "HazardAtSideLane",
    "AccidentTwoWays",
    "BlockedIntersection",
    "HighwayExit",
    "MergerIntoSlowTraffic",
    "NonSignalizedJunctionLeftTurn",
    "NonSignalizedJunctionRightTurn",
    "PedestrianCrossing",
    "RedLightWithoutLeadVehicle",
    "VehicleTurningRoutePedestrian",
)
COMMANDS = {
    0: "unknown",
    1: "turn_left",
    2: "turn_right",
    3: "straight",
    4: "follow_lane",
    5: "change_lane_left",
    6: "change_lane_right",
}
MANEUVERS = {
    "keep_lane", "turn_left", "turn_right", "change_lane_left",
    "change_lane_right", "stop_wait",
}
INTERACTIONS = {"none", "follow", "yield", "blocked", "caution"}
PROMPT_VERSION = "p7_ar_intent_cpu_probe_v1"
PROMPT_V2 = "p7_ar_intent_cpu_probe_v2"


def prompt(command: str, speed: float, version: str = PROMPT_VERSION) -> str:
    if version == PROMPT_V2:
        return (
            "The three images are synchronized CARLA driving views in this order: "
            "front, front-left, front-right. Identify candidate ego ROUTES for the "
            "next 10 meters, not immediate throttle or brake actions. A route is a "
            "distinct visible path through the road geometry that is consistent "
            "with the navigation instruction. List up to three such routes; one "
            "route is enough if no distinct alternative is visible. Do not invent "
            "lanes, turns, exits, or actors. Waiting or yielding on the same route "
            "does not create another route: represent that with the interaction "
            "field. Use stop_wait as the only intent only when no traversable "
            "route is visible. Return ONLY one JSON object with key 'intents'. "
            "For each item, maneuver must be exactly one of keep_lane, turn_left, "
            "turn_right, change_lane_left, change_lane_right, stop_wait; "
            "interaction must be exactly one of none, follow, yield, blocked, "
            "caution; confidence must be a number from 0 to 1. Do not use "
            "interaction words as maneuvers. No explanation or markdown. "
            f"Navigation: {command}. Ego speed: {speed:.1f} m/s."
        )
    if version != PROMPT_VERSION:
        raise ValueError(f"unknown prompt version: {version}")
    return (
        "The three images are synchronized CARLA driving views in this order: "
        "front, front-left, front-right. Based only on these images and the "
        "navigation instruction, list distinct plausible near-term ego maneuvers. "
        "A maneuver may be geometrically possible even if it currently requires yielding. "
        "Do not invent lanes, turns, or actors. Prefer fewer supported maneuvers. "
        "Return ONLY one JSON object with key 'intents' containing at most three items. "
        "Each item must contain 'maneuver' (keep_lane, turn_left, turn_right, "
        "change_lane_left, change_lane_right, stop_wait), 'interaction' (none, "
        "follow, yield, blocked, caution), and numeric 'confidence' between 0 and 1. "
        "No explanation or markdown. "
        f"Navigation: {command}. Ego speed: {speed:.1f} m/s."
    )


def parse_intents(text: str) -> tuple[list[dict], str]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], f"json_error: {exc.msg}"
    if not isinstance(value, dict) or not isinstance(value.get("intents"), list):
        return [], "schema_error: missing intents list"
    intents = value["intents"]
    if len(intents) > 3:
        return [], "schema_error: too many intents"
    seen: set[str] = set()
    for item in intents:
        if not isinstance(item, dict):
            return [], "schema_error: intent is not an object"
        maneuver = item.get("maneuver")
        interaction = item.get("interaction")
        confidence = item.get("confidence")
        if maneuver not in MANEUVERS or interaction not in INTERACTIONS:
            return [], "schema_error: invalid vocabulary"
        if maneuver in seen:
            return [], "schema_error: duplicate maneuver"
        if isinstance(confidence, bool) or not isinstance(confidence, (float, int)):
            return [], "schema_error: missing numeric confidence"
        if not 0 <= confidence <= 1:
            return [], "schema_error: confidence out of range"
        seen.add(maneuver)
    return intents, "ok"


def sample_paths(data_root: Path) -> list[tuple[str, Path]]:
    selected = []
    for scenario in SCENARIOS:
        routes = sorted((data_root / scenario).glob("*/rgb"))
        if not routes:
            raise FileNotFoundError(f"no RGB routes for {scenario}")
        images = sorted(routes[0].glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"no JPEG images in {routes[0]}")
        selected.append((scenario, images[len(images) // 2]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--official-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--prompt-version", choices=(PROMPT_VERSION, PROMPT_V2), default=PROMPT_VERSION
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.official_code / "src"))
    import torch
    import transformers
    from PIL import Image
    from qwen_drive import QwenDriveForPlanning

    torch.set_num_threads(16)
    torch.set_num_interop_threads(2)
    warnings.filterwarnings("ignore", message="numpy.core.numeric is deprecated")

    started = time.monotonic()
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cpu").eval()
    load_seconds = time.monotonic() - started
    print(f"loaded VLM on CPU in {load_seconds:.1f}s", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.output.open("w", encoding="utf-8") as output:
        for index, (scenario, image_path) in enumerate(sample_paths(args.data_root), 1):
            meta_path = image_path.parent.parent / "metas" / (image_path.stem + ".pkl")
            with lzma.open(meta_path, "rb") as stream:
                meta = pickle.load(stream)
            command_id = int(meta["next_commands"][0])
            command = COMMANDS.get(command_id, "unknown")
            speed = float(meta["speed"])
            with Image.open(image_path) as strip:
                strip = strip.convert("RGB")
                width = strip.width // 3
                left = strip.crop((0, 0, width, strip.height))
                front = strip.crop((width, 0, 2 * width, strip.height))
                right = strip.crop((2 * width, 0, 3 * width, strip.height))
            question = prompt(command, speed, args.prompt_version)
            t0 = time.monotonic()
            try:
                result = model.generate_text(
                    [front, left, right], question, max_new_tokens=args.max_new_tokens,
                )
                raw = result.text or ""
                intents, status = parse_intents(raw)
                error = None
            except Exception as exc:
                raw, intents = "", []
                status, error = "inference_error", f"{type(exc).__name__}: {exc}"
            row = {
                "index": index, "scenario": scenario, "image": str(image_path.resolve()),
                "command_id": command_id, "command": command, "speed_mps": speed,
                "prompt_version": args.prompt_version, "raw_text": raw,
                "intents": intents, "status": status, "error": error,
                "seconds": round(time.monotonic() - t0, 3),
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            rows.append(row)
            print(f"{index}/10 {scenario}: {status}, {row['seconds']:.1f}s, "
                  f"intents={len(intents)}", flush=True)

    durations = [row["seconds"] for row in rows]
    summary = {
        "prompt_version": args.prompt_version,
        "model": str(args.model),
        "official_code": str(args.official_code),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": "cpu", "dtype": "float32", "attention": "sdpa",
        "max_new_tokens": args.max_new_tokens,
        "model_load_seconds": round(load_seconds, 3),
        "samples": len(rows),
        "schema_ok": sum(row["status"] == "ok" for row in rows),
        "nonempty_intents": sum(bool(row["intents"]) for row in rows),
        "latency_median_seconds": statistics.median(durations),
        "latency_max_seconds": max(durations),
        "latency_seconds": durations,
    }
    summary_path = args.output.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.output} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
