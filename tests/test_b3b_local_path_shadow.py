import numpy as np
from copy import deepcopy

from lead.common.pid_controller import LateralPIDController
from lead.data_loader.future_actor_cache import FUTURE_STEPS, FutureActorFrame
from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.tfv6.local_path_shadow import (
    encode_intent_probability_map,
    local_path_variants,
    score_local_paths,
)
from scripts.p4.analyze_b3b_local_path_shadow import decode_intent_probability_map


class FakeLateralController:
    def __init__(self):
        self.calls = 0

    def step(self, route, current_speed, ego_location, ego_rotation, *, sensor_agent_steer_correction):
        self.calls += 1
        return round(float(route[0, 1]), 3)


def _actors_at_x10():
    return FutureActorFrame(
        positions=np.tile(np.array([[[10.0, 0.0]]], dtype=np.float32), (1, FUTURE_STEPS, 1)),
        yaws=np.zeros((1, FUTURE_STEPS), dtype=np.float32),
        extents=np.array([[0.2, 0.2, 0.5]], dtype=np.float32),
        z=np.zeros(1, dtype=np.float32),
        valid=np.ones((1, FUTURE_STEPS), dtype=bool),
        actor_ids=np.array([1], dtype=np.int64),
        class_ids=np.array([1], dtype=np.uint8),
        ego_extent=np.array([1.0, 0.5, 0.75], dtype=np.float32),
    )


def test_shadow_scores_local_paths_without_mutating_controller():
    raw = np.stack((np.arange(1, 31), np.zeros(30)), axis=-1).astype(np.float32)
    routes = np.concatenate((raw[None], local_path_variants(raw)), axis=0)
    controller = FakeLateralController()
    record = score_local_paths(
        routes, np.zeros((5, 30), dtype=np.float32), _actors_at_x10(), controller,
        current_speed_mps=5.0, raw_target_speed_mps=5.0, baseline_steer=0.0,
    )
    assert controller.calls == 0
    assert record["shadow_only"]
    assert record["raw_predicted_collision"]
    assert record["would_switch"]
    assert record["would_select_index"] in (1, 2, 3, 4)
    assert len(record["candidates"]) == 5
    assert record["candidates"][0]["predicted_ttc_s"] is not None
    assert record["candidates"][0]["predicted_collision_actor_id"] == 1
    assert len(record["candidates"][0]["corridor_per_point"]) == 30
    assert record["predicted_actors"][0]["id"] == 1


def test_shadow_never_proposes_switch_below_speed_floor():
    raw = np.stack((np.arange(1, 31), np.zeros(30)), axis=-1).astype(np.float32)
    routes = np.concatenate((raw[None], local_path_variants(raw)), axis=0)
    record = score_local_paths(
        routes, np.zeros((5, 30), dtype=np.float32), _actors_at_x10(),
        FakeLateralController(), current_speed_mps=0.0,
        raw_target_speed_mps=0.0, baseline_steer=0.0,
    )
    assert not record["would_switch"]


def test_actual_pid_shadow_matches_baseline_without_advancing_state():
    raw = np.stack((np.arange(1, 31), np.zeros(30)), axis=-1).astype(np.float32)
    routes = np.concatenate((raw[None], local_path_variants(raw)), axis=0)
    controller = LateralPIDController(ClosedLoopConfig())
    snapshot = deepcopy(controller)
    baseline = controller.step(raw, 5.0, 0.0, 0.0)
    baseline_history = controller._window.copy()
    record = score_local_paths(
        routes, np.zeros((5, 30), dtype=np.float32), _actors_at_x10(),
        snapshot, current_speed_mps=5.0,
        raw_target_speed_mps=5.0, baseline_steer=baseline,
    )
    assert record["pid_snapshot_raw_steer"] == baseline
    assert controller._window == baseline_history


def test_intent_probability_map_roundtrip():
    probability = np.array([[0.0, 0.25, 0.5], [0.75, 1.0, 0.3]], dtype=np.float32)
    encoded = encode_intent_probability_map(probability)
    decoded = decode_intent_probability_map({"intent_probability_map": encoded})
    assert encoded["fraction_above_0_5"] == 2 / 6
    np.testing.assert_allclose(decoded, probability, atol=1 / 255)
