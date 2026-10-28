from lead.tfv6.semantic_intent_slots import prepare_planner_intents


def proposal(*maneuvers, signal="none"):
    return {
        "scene_rules": {
            "signal": signal, "stop_sign": "no",
            "pedestrian_conflict": "no", "vehicle_conflict": "no",
        },
        "intents": [
            {"maneuver": m, "obligations": [], "confidence": 0.7}
            for m in maneuvers
        ],
    }


def test_multiple_vlm_intents_reach_distinct_query_slots_without_topology_gate():
    packed = prepare_planner_intents(
        proposal("keep_lane", "change_lane_left", "change_lane_right"),
        "follow_lane", capacity=4,
    )
    assert [s["maneuver"] for s in packed["slots"]] == [
        "keep_lane", "change_lane_left", "change_lane_right", None
    ]
    assert packed["valid"] == [True, True, True, False]
    assert not packed["fallback_required"]
    assert packed["validity_scope"] == "schema_and_coarse_navigation_only"


def test_red_light_adds_obligation_without_deleting_route():
    packed = prepare_planner_intents(proposal("turn_right", signal="red"), "turn_right")
    assert packed["valid"][0]
    assert packed["slots"][0]["obligations"] == ["stop_for_red"]


def test_obvious_navigation_conflict_is_excluded():
    packed = prepare_planner_intents(proposal("turn_left", "turn_right"), "turn_left")
    assert packed["valid"][:2] == [True, False]
    assert "navigation_conflict:turn_right" in packed["rejected"]


def test_unknown_navigation_uses_fallback():
    packed = prepare_planner_intents(proposal("keep_lane"), "unknown")
    assert packed["fallback_required"]


def test_slot_identity_does_not_follow_vlm_list_order():
    packed = prepare_planner_intents(
        proposal("change_lane_right", "keep_lane", "change_lane_left"),
        "follow_lane", capacity=3,
    )
    assert [s["maneuver"] for s in packed["slots"]] == [
        "keep_lane", "change_lane_left", "change_lane_right"
    ]
