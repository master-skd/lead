"""Conservative, independent validation of proposed P7 route intents.

The caller must supply current-frame evidence from a source *other than* the
VLM proposal (for example, an independently validated perception or map
adapter).  Neither VLM prose nor CARLA future/expert labels are admissible as
evidence.  Missing evidence fails closed; this module does not infer road
geometry from the maneuver name or confidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MANEUVERS = frozenset({
    "keep_lane", "change_lane_left", "change_lane_right", "turn_left", "turn_right",
})
GEOMETRY_FACTS = (
    "exists", "same_direction", "boundary_legal", "traversable", "nav_compatible",
)
CONFLICT_FACTS = (
    "pedestrian_conflict", "vehicle_conflict", "lead_vehicle", "obstacle",
)
OBLIGATION_FOR_FACT = {
    "pedestrian_conflict": "yield_pedestrian",
    "vehicle_conflict": "yield_vehicle",
    "lead_vehicle": "follow_lead",
    "obstacle": "slow_for_obstacle",
}
OBLIGATION_ORDER = (
    "stop_for_red", "stop_at_sign", "yield_pedestrian", "yield_vehicle",
    "follow_lead", "slow_for_obstacle",
)


def _boolean_fact(source: Mapping[str, Any], key: str, reasons: list[str]) -> bool | None:
    value = source.get(key)
    if type(value) is not bool:
        reasons.append(f"{key}:unknown")
        return None
    return value


def validate_intents(
    proposal: Mapping[str, Any] | None,
    evidence: Mapping[str, Any],
    *,
    capacity: int = 6,
) -> dict[str, Any]:
    """Return fixed K slots, validity, obligations, uncertainty and fallback.

    ``evidence`` contract::

      {"signal": "red|yellow|green|none", "stop_sign": bool,
       "maneuvers": {"keep_lane": {
           "exists": bool, "same_direction": bool,
           "boundary_legal": bool, "traversable": bool,
           "nav_compatible": bool, "pedestrian_conflict": bool,
           "vehicle_conflict": bool, "lead_vehicle": bool,
           "obstacle": bool}}}

    All fields must be established for the *current frame* independently of
    the proposal.  ``False`` on any geometry fact rejects the route.  Unknown
    signal, stop sign or route conflict facts also reject it.  A red light or
    stop sign keeps the geometry but adds a mandatory stop obligation.
    """
    if capacity < 1:
        raise ValueError("capacity must be positive")
    proposed = proposal.get("intents", []) if isinstance(proposal, Mapping) else []
    if not isinstance(proposed, list):
        proposed = []
    routes = evidence.get("maneuvers", {})
    if not isinstance(routes, Mapping):
        routes = {}

    slots: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in proposed:
        if len(slots) >= capacity:
            break
        if not isinstance(item, Mapping):
            continue
        maneuver = item.get("maneuver")
        if not isinstance(maneuver, str) or maneuver not in MANEUVERS or maneuver in seen:
            continue
        seen.add(maneuver)
        route = routes.get(maneuver, {})
        if not isinstance(route, Mapping):
            route = {}
        reasons: list[str] = []
        geometry = {key: _boolean_fact(route, key, reasons) for key in GEOMETRY_FACTS}
        for key, value in geometry.items():
            if value is False:
                reasons.append(f"{key}:false")

        signal = evidence.get("signal")
        if not isinstance(signal, str) or signal not in {"red", "yellow", "green", "none"}:
            reasons.append("signal:unknown")
        elif signal == "yellow":
            # The schema has no stop-if-safe/yellow obligation yet.
            reasons.append("signal:yellow_requires_decision")
        stop_sign = _boolean_fact(evidence, "stop_sign", reasons)
        conflicts = {key: _boolean_fact(route, key, reasons) for key in CONFLICT_FACTS}
        uncertain = any(reason.endswith(":unknown") for reason in reasons)
        valid = not reasons

        mandatory: set[str] = set()
        if signal == "red":
            mandatory.add("stop_for_red")
        if stop_sign is True:
            mandatory.add("stop_at_sign")
        for key, obligation in OBLIGATION_FOR_FACT.items():
            if conflicts[key] is True:
                mandatory.add(obligation)

        proposed_obligations = item.get("obligations", [])
        if not isinstance(proposed_obligations, list):
            proposed_obligations = []
        proposed_obligations = {o for o in proposed_obligations if isinstance(o, str)}
        proposed_obligations &= set(OBLIGATION_ORDER)
        slots.append({
            "maneuver": maneuver,
            "valid": valid,
            "uncertain": uncertain,
            "obligations": [o for o in OBLIGATION_ORDER if o in mandatory],
            "reasons": reasons,
            "vlm_obligation_disagreement": sorted(proposed_obligations ^ mandatory),
        })

    while len(slots) < capacity:
        slots.append({
            "maneuver": None, "valid": False, "uncertain": False,
            "obligations": [], "reasons": ["padding"],
            "vlm_obligation_disagreement": [],
        })
    return {
        "slots": slots,
        "valid": [slot["valid"] for slot in slots],
        "uncertain": [slot["uncertain"] for slot in slots],
        "obligations": [slot["obligations"] for slot in slots],
        "fallback_required": not any(slot["valid"] for slot in slots),
    }
