"""Current-frame BEV adapter must preserve missing topology as unknown."""

from types import SimpleNamespace

import numpy as np

from lead.common.constants import TransfuserBEVSemanticClass as C
from lead.tfv6.semantic_intent_bev_evidence import validate_with_bev


CONFIG = SimpleNamespace(pixels_per_meter=1, min_x_meter=0, min_y_meter=0)


def test_road_pixels_do_not_certify_a_lane_change():
    bev = np.full((20, 20), int(C.ROAD), dtype=np.uint8)
    route = np.array([[2, 2], [3, 2], [4, 2]], dtype=np.float32)
    proposal = {"intents": [{"maneuver": "change_lane_left", "obligations": []}]}
    result = validate_with_bev(
        proposal, bev, {"change_lane_left": route}, CONFIG,
        signal="none", stop_sign=False,
    )
    assert result["fallback_required"]
    assert result["uncertain"][0]
    assert "same_direction:unknown" in result["slots"][0]["reasons"]


def test_independent_topology_and_bev_can_admit_branch_with_obligations():
    bev = np.full((20, 20), int(C.ROAD), dtype=np.uint8)
    bev[2, 3] = int(C.VEHICLE)
    route = np.array([[2, 2], [3, 2], [4, 2]], dtype=np.float32)
    proposal = {"intents": [{"maneuver": "keep_lane", "obligations": []}]}
    result = validate_with_bev(
        proposal, bev, {"keep_lane": route}, CONFIG,
        topology={"keep_lane": {
            "exists": True, "same_direction": True,
            "boundary_legal": True, "nav_compatible": True,
        }}, signal="red", stop_sign=False,
    )
    assert result["valid"][0]
    assert result["obligations"][0] == [
        "stop_for_red", "yield_vehicle", "follow_lead"
    ]


def test_off_map_route_is_rejected():
    bev = np.full((20, 20), int(C.ROAD), dtype=np.uint8)
    route = np.array([[2, 2], [30, 2]], dtype=np.float32)
    proposal = {"intents": [{"maneuver": "keep_lane", "obligations": []}]}
    result = validate_with_bev(
        proposal, bev, {"keep_lane": route}, CONFIG,
        topology={"keep_lane": {
            "exists": True, "same_direction": True,
            "boundary_legal": True, "nav_compatible": True,
        }}, signal="none", stop_sign=False,
    )
    assert result["fallback_required"]
    assert "traversable:false" in result["slots"][0]["reasons"]
