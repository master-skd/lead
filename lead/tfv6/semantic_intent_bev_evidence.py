"""Current-frame LEAD BEV evidence for the P7 intent validator.

This is a *post-route* adapter: candidate routes and the predicted BEV semantic
map come from the current sensor frame.  It cannot establish lane direction,
marking colour, or turn-lane legality.  Those facts must come from a separate
current-frame topology source and remain unknown when unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from lead.common.constants import TransfuserBEVSemanticClass as C
from lead.tfv6.semantic_intent_validator import MANEUVERS, validate_intents

ROADLIKE = {int(C.ROAD), int(C.LANE_MARKERS)}
DYNAMIC = {int(C.VEHICLE), int(C.WALKER), int(C.SPECIAL_VEHICLE), int(C.BIKER)}
STATIC_BLOCK = {int(C.OBSTACLE), int(C.PARKING_VEHICLE)}
POSSIBLE_ROAD = ROADLIKE | DYNAMIC | STATIC_BLOCK


def _route_classes(
    bev_classes: np.ndarray, route_xy: np.ndarray, config: Any,
) -> tuple[np.ndarray, bool]:
    if route_xy.ndim != 2 or route_xy.shape[1] != 2 or not len(route_xy):
        raise ValueError("route_xy must have shape (N, 2) with N > 0")
    if not np.isfinite(route_xy).all():
        return np.empty(0, dtype=np.int64), False
    ppm = float(config.pixels_per_meter)
    col = np.rint((route_xy[:, 0] - float(config.min_x_meter)) * ppm).astype(int)
    row = np.rint((route_xy[:, 1] - float(config.min_y_meter)) * ppm).astype(int)
    h, w = bev_classes.shape
    inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    if not inside.all():
        return np.empty(0, dtype=np.int64), False
    return bev_classes[row, col], True


def evidence_from_bev(
    bev_classes: np.ndarray,
    routes_by_maneuver: Mapping[str, np.ndarray],
    config: Any,
    *,
    topology: Mapping[str, Mapping[str, bool | None]] | None = None,
    signal: str | None = None,
    stop_sign: bool | None = None,
) -> dict[str, Any]:
    """Build partial independent evidence for current-frame candidate routes.

    ``topology`` must be produced independently of the VLM proposal.  It may
    establish ``exists``, ``same_direction``, ``boundary_legal`` and
    ``nav_compatible``; missing values stay unknown.  ``signal`` and
    ``stop_sign`` require an independently confirmed applicable control.  BEV
    absence is not proof that no traffic light or stop sign applies.

    BEV semantic classes contain neither marking colour nor lane direction;
    a path through ROAD pixels therefore never proves a lane change legal.
    """
    bev = np.asarray(bev_classes)
    if bev.ndim != 2 or not np.issubdtype(bev.dtype, np.integer):
        raise ValueError("bev_classes must be a (H, W) integer class map")
    topology = topology or {}
    maneuvers: dict[str, dict[str, bool | None]] = {}
    for maneuver, points in routes_by_maneuver.items():
        if maneuver not in MANEUVERS:
            continue
        independent = topology.get(maneuver, {})
        if not isinstance(independent, Mapping):
            independent = {}
        facts = {key: independent.get(key) for key in (
            "exists", "same_direction", "boundary_legal", "nav_compatible"
        )}
        classes, in_bounds = _route_classes(bev, np.asarray(points), config)
        if not in_bounds:
            facts.update({
                "traversable": False, "pedestrian_conflict": None,
                "vehicle_conflict": None, "lead_vehicle": None,
                "obstacle": None,
            })
        else:
            roadlike = np.isin(classes, list(POSSIBLE_ROAD))
            static = np.isin(classes, list(STATIC_BLOCK))
            vehicle = np.isin(classes, [int(C.VEHICLE), int(C.SPECIAL_VEHICLE)])
            facts.update({
                "traversable": bool(roadlike.all() and not static.any()),
                "pedestrian_conflict": bool(np.isin(classes, [int(C.WALKER), int(C.BIKER)]).any()),
                "vehicle_conflict": bool(vehicle.any()),
                "lead_vehicle": bool(vehicle.any() and maneuver == "keep_lane"),
                "obstacle": bool(static.any()),
            })
        maneuvers[maneuver] = facts
    return {
        "signal": signal, "stop_sign": stop_sign,
        "maneuvers": maneuvers,
        "provenance": "current_frame_predicted_bev_plus_independent_topology",
    }


def validate_with_bev(
    proposal: Mapping[str, Any] | None,
    bev_classes: np.ndarray,
    routes_by_maneuver: Mapping[str, np.ndarray],
    config: Any,
    *,
    topology: Mapping[str, Mapping[str, bool | None]] | None = None,
    signal: str | None = None,
    stop_sign: bool | None = None,
    capacity: int = 6,
) -> dict[str, Any]:
    """Connect predicted BEV evidence to the fixed-slot intent gate."""
    evidence = evidence_from_bev(
        bev_classes, routes_by_maneuver, config,
        topology=topology, signal=signal, stop_sign=stop_sign,
    )
    result = validate_intents(proposal, evidence, capacity=capacity)
    result["evidence"] = evidence
    return result
