"""Behavioral checks for the P7 independent evidence gate."""

from lead.tfv6.semantic_intent_validator import validate_intents


def route(**changes):
    facts = {
        "exists": True, "same_direction": True, "boundary_legal": True,
        "traversable": True, "nav_compatible": True,
        "pedestrian_conflict": False, "vehicle_conflict": False,
        "lead_vehicle": False, "obstacle": False,
    }
    facts.update(changes)
    return facts


def proposal(*maneuvers):
    return {"intents": [
        {"maneuver": m, "obligations": [], "confidence": 1.0}
        for m in maneuvers
    ]}


def test_rejects_oncoming_lane_even_with_high_vlm_confidence():
    evidence = {"signal": "none", "stop_sign": False, "maneuvers": {
        "keep_lane": route(), "change_lane_left": route(same_direction=False),
    }}
    result = validate_intents(proposal("keep_lane", "change_lane_left"), evidence)
    assert result["valid"] == [True, False, False, False, False, False]
    assert "same_direction:false" in result["slots"][1]["reasons"]
    assert not result["fallback_required"]


def test_keeps_route_and_forces_stop_and_yield_obligations():
    evidence = {"signal": "red", "stop_sign": True, "maneuvers": {
        "turn_right": route(pedestrian_conflict=True, vehicle_conflict=True),
    }}
    result = validate_intents(proposal("turn_right"), evidence)
    assert result["valid"][0]
    assert result["obligations"][0] == [
        "stop_for_red", "stop_at_sign", "yield_pedestrian", "yield_vehicle"
    ]
    assert "stop_for_red" in result["slots"][0]["vlm_obligation_disagreement"]


def test_missing_independent_facts_fail_closed():
    result = validate_intents(proposal("keep_lane"), {"maneuvers": {
        "keep_lane": {"exists": True},
    }})
    assert not result["valid"][0]
    assert result["uncertain"][0]
    assert result["fallback_required"]


def test_blocks_unknown_navigation_and_yellow_signal():
    evidence = {"signal": "none", "stop_sign": False, "maneuvers": {
        "change_lane_right": route(nav_compatible=False),
    }}
    result = validate_intents(proposal("change_lane_right"), evidence)
    assert result["fallback_required"]
    evidence["signal"] = "yellow"
    evidence["maneuvers"]["change_lane_right"] = route()
    result = validate_intents(proposal("change_lane_right"), evidence)
    assert result["fallback_required"]
    assert "signal:yellow_requires_decision" in result["slots"][0]["reasons"]
