"""Turn Qwen-Drive proposals into P7 planner query slots.

This is a *semantic* gate, not an independent road-legality verifier.  It keeps
the VLM's branch identity and obligations intact for the Query adapter.
"""

from __future__ import annotations

import math
import json
import re
from collections.abc import Mapping
from typing import Any

from lead.tfv6.semantic_intent_validator import MANEUVERS, OBLIGATION_ORDER

SIGNALS = {"red", "yellow", "green", "none", "uncertain"}
TRISTATE = {"yes", "no", "uncertain"}
MANEUVER_ORDER = (
    "keep_lane", "change_lane_left", "change_lane_right", "turn_left", "turn_right",
)
MANEUVER_RANK = {maneuver: index for index, maneuver in enumerate(MANEUVER_ORDER)}
MANEUVER_ID = {maneuver: index + 1 for index, maneuver in enumerate(MANEUVER_ORDER)}
SIGNAL_ID = {"uncertain": 0, "none": 1, "red": 2, "yellow": 3, "green": 4}
FLAG_ID = {"uncertain": 0, "no": 1, "yes": 2}
NAV_ALLOWED = {
    "follow_lane": MANEUVERS - {"turn_left", "turn_right"},
    "straight": MANEUVERS - {"turn_left", "turn_right"},
    "turn_left": {"keep_lane", "turn_left", "change_lane_left"},
    "turn_right": {"keep_lane", "turn_right", "change_lane_right"},
    "change_lane_left": {"keep_lane", "change_lane_left"},
    "change_lane_right": {"keep_lane", "change_lane_right"},
}


def prepare_planner_intents(
    proposal: Mapping[str, Any] | None,
    navigation: str,
    *,
    capacity: int = 6,
) -> dict[str, Any]:
    """Pack VLM maneuvers/obligations for one intent vector per valid slot.

    The result's ``valid`` means schema and coarse navigation compatibility
    only. It does *not* assert that a lane exists or is legally traversable.
    Unknown/failed responses request the separately validated LEAD fallback.
    """
    if capacity < 1:
        raise ValueError("capacity must be positive")
    if not isinstance(navigation, str) or navigation not in NAV_ALLOWED or not isinstance(proposal, Mapping):
        return _empty(capacity, "missing_proposal_or_navigation")
    rules, raw = proposal.get("scene_rules"), proposal.get("intents")
    if not isinstance(rules, Mapping) or not isinstance(raw, list):
        return _empty(capacity, "invalid_schema")
    if (not isinstance(rules.get("signal"), str) or rules.get("signal") not in SIGNALS
            or any(not isinstance(rules.get(k), str) or rules.get(k) not in TRISTATE for k in (
                "stop_sign", "pedestrian_conflict", "vehicle_conflict"
            ))):
        return _empty(capacity, "invalid_schema")

    allowed = NAV_ALLOWED[navigation]
    slots: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejected: list[str] = []
    for item in raw:
        if len(slots) >= capacity:
            rejected.append("capacity_exceeded")
            break
        if not isinstance(item, Mapping):
            rejected.append("invalid_item")
            continue
        maneuver = item.get("maneuver")
        obligations = item.get("obligations")
        confidence = item.get("confidence")
        if (not isinstance(maneuver, str) or maneuver not in MANEUVERS
                or not isinstance(obligations, list)
                or any(not isinstance(o, str) or o not in OBLIGATION_ORDER for o in obligations)
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            rejected.append("invalid_item")
            continue
        if maneuver in seen:
            rejected.append("duplicate_maneuver")
            continue
        seen.add(maneuver)
        if maneuver not in allowed:
            rejected.append(f"navigation_conflict:{maneuver}")
            continue
        duties = set(obligations)
        if rules["signal"] == "red":
            duties.add("stop_for_red")
        if rules["stop_sign"] == "yes":
            duties.add("stop_at_sign")
        if rules["pedestrian_conflict"] == "yes":
            duties.add("yield_pedestrian")
        # Vehicle conflict is often route-specific; preserve the VLM's own
        # per-branch obligation rather than forcing it onto every maneuver.
        slots.append({
            "maneuver": maneuver,
            "obligations": [o for o in OBLIGATION_ORDER if o in duties],
            "valid": True,
            "uncertain": "uncertain" in rules.values(),
        })
    slots.sort(key=lambda slot: MANEUVER_RANK[slot["maneuver"]])
    while len(slots) < capacity:
        slots.append({"maneuver": None, "obligations": [], "valid": False, "uncertain": False})
    return {
        "slots": slots,
        "scene_rules": dict(rules),
        "valid": [s["valid"] for s in slots],
        "fallback_required": not any(s["valid"] for s in slots),
        "rejected": rejected,
        "validity_scope": "schema_and_coarse_navigation_only",
    }


def _empty(capacity: int, reason: str) -> dict[str, Any]:
    return {
        "slots": [
            {"maneuver": None, "obligations": [], "valid": False, "uncertain": False}
            for _ in range(capacity)
        ],
        "valid": [False] * capacity,
        "fallback_required": True,
        "rejected": [reason],
        "validity_scope": "schema_and_coarse_navigation_only",
        "scene_rules": {
            "signal": "uncertain", "stop_sign": "uncertain",
            "pedestrian_conflict": "uncertain", "vehicle_conflict": "uncertain",
        },
    }


def slots_to_tensors(packed_batch: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Convert a batch of packed VLM intents to the Planner's tensor contract."""
    import torch

    if not packed_batch:
        raise ValueError("packed_batch must not be empty")
    capacity = len(packed_batch[0]["slots"])
    if any(len(item["slots"]) != capacity for item in packed_batch):
        raise ValueError("all samples must use the same K")
    maneuver = torch.zeros(len(packed_batch), capacity, dtype=torch.long)
    obligation = torch.zeros(len(packed_batch), capacity, len(OBLIGATION_ORDER))
    scene = torch.zeros(len(packed_batch), 4, dtype=torch.long)
    valid = torch.zeros(len(packed_batch), capacity, dtype=torch.bool)
    for b, item in enumerate(packed_batch):
        rules = item["scene_rules"]
        scene[b] = torch.tensor([
            SIGNAL_ID[rules["signal"]], FLAG_ID[rules["stop_sign"]],
            FLAG_ID[rules["pedestrian_conflict"]], FLAG_ID[rules["vehicle_conflict"]],
        ])
        for k, slot in enumerate(item["slots"]):
            if not slot["valid"]:
                continue
            valid[b, k] = True
            maneuver[b, k] = MANEUVER_ID[slot["maneuver"]]
            for obligation_index, name in enumerate(OBLIGATION_ORDER):
                obligation[b, k, obligation_index] = float(name in slot["obligations"])
    return {
        "semantic_intent_maneuver_ids": maneuver,
        "semantic_intent_obligations": obligation,
        "semantic_intent_scene_rules": scene,
        "semantic_intent_valid": valid,
    }


def parse_service_result(result: Mapping[str, Any]):
    """Use the same text parser for online replies and offline JSON records."""
    raw = result.get("raw_text", "")
    if isinstance(raw, str) and not result.get("error"):
        candidate = raw.strip()
        fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", candidate, flags=re.DOTALL)
        if fence:
            candidate = fence.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return None


def service_results_to_tensors(
    results: list[Mapping[str, Any]], navigation: list[str], *, capacity: int,
) -> dict[str, Any]:
    """Convert live HF VQA replies to P7 tensors with explicit LEAD fallback.

    Failed or empty replies reserve only slot 0. The accompanying fallback mask
    makes the Planner skip its semantic residual for those rows, so slot 0 uses
    the copied posttrain LEAD route query rather than an invented VLM branch.
    """
    import torch

    if len(results) != len(navigation):
        raise ValueError("results/navigation length mismatch")
    packed_batch = []
    fallback = []
    for result, command in zip(results, navigation, strict=True):
        proposal = parse_service_result(result)
        packed = prepare_planner_intents(proposal, command, capacity=capacity)
        empty = packed["fallback_required"]
        if empty:
            packed["slots"][0] = {
                "maneuver": "keep_lane", "obligations": [],
                "valid": True, "uncertain": True,
            }
            packed["valid"][0] = True
        packed_batch.append(packed)
        fallback.append(empty)
    tensors = slots_to_tensors(packed_batch)
    tensors["semantic_intent_fallback"] = torch.tensor(fallback, dtype=torch.bool)
    return tensors
