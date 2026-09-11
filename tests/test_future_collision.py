import numpy as np

from lead.data_loader.future_actor_cache import (
    FUTURE_STEPS,
    FutureActorFrame,
    parse_future_actor_frame,
)
from lead.tfv6.future_collision import (
    future_collision_label,
    interpolate_route_by_distance,
    speed_profile_distances,
)


def _actors(position, yaw=0.0, valid=None, actor_id=7):
    if valid is None:
        valid = np.ones((1, FUTURE_STEPS), dtype=bool)
    positions = np.repeat(np.asarray(position, dtype=np.float32)[None, None], FUTURE_STEPS, axis=1)
    return FutureActorFrame(
        positions=positions,
        yaws=np.full((1, FUTURE_STEPS), yaw, dtype=np.float32),
        extents=np.asarray([[1.0, 0.5, 0.5]], dtype=np.float32),
        z=np.zeros(1, dtype=np.float32),
        valid=valid,
        actor_ids=np.asarray([actor_id]),
        class_ids=np.asarray([1], dtype=np.uint8),
        ego_extent=np.asarray([1.0, 0.5, 0.5], dtype=np.float32),
    )


def test_obb_collision_reports_first_time_and_actor():
    ego = np.stack([np.arange(1, 9), np.zeros(8)], axis=1).astype(np.float32)
    label = future_collision_label(ego, np.zeros(8), _actors([4.0, 0.0]), safety_margin_m=0.0)
    assert label.collision
    assert label.ttc_s == 0.5
    assert label.collision_step == 1
    assert label.actor_id == 7


def test_rotated_separated_boxes_do_not_collide():
    ego = np.zeros((8, 2), dtype=np.float32)
    label = future_collision_label(
        ego, np.zeros(8), _actors([0.0, 3.0], yaw=np.pi / 4), safety_margin_m=0.0,
    )
    assert not label.collision


def test_invalid_future_tail_is_not_treated_as_ground_truth():
    valid = np.zeros((1, 8), dtype=bool)
    valid[:, :2] = True
    actors = _actors([0.0, 0.0], valid=valid)
    actors.positions[:, :2] = 100.0
    ego = np.zeros((8, 2), dtype=np.float32)
    assert not future_collision_label(ego, np.zeros(8), actors).collision


def test_class_filter_can_exclude_static_collision():
    actors = _actors([0.0, 0.0])
    actors.class_ids[:] = 3
    ego = np.zeros((8, 2), dtype=np.float32)
    assert future_collision_label(ego, np.zeros(8), actors).collision
    assert not future_collision_label(
        ego, np.zeros(8), actors, include_class_ids=(1, 2),
    ).collision


def test_parser_samples_20hz_future_and_keeps_partial_mask():
    future = np.stack([np.arange(13), np.zeros(13)], axis=1).astype(np.float32)
    boxes = [
        {"class": "ego_car", "extent": [2.0, 1.0, 0.5], "position": [0, 0, 0]},
        {
            "class": "car", "id": 12, "extent": [1.0, 0.5, 0.5],
            "position": [0, 0, 0], "yaw": 0.0,
            "future_positions": future, "future_yaws": np.zeros(13),
        },
        {"class": "traffic_light", "id": 99, "extent": [1, 1, 1], "position": [2, 0, 0]},
    ]
    frame = parse_future_actor_frame(boxes)
    assert frame.num_actors == 1
    np.testing.assert_array_equal(frame.valid[0], [True, True, False, False, False, False, False, False])
    np.testing.assert_allclose(frame.positions[0, :2, 0], [5.0, 10.0])
    np.testing.assert_allclose(frame.ego_extent, [2.0, 1.0, 0.5])


def test_route_interpolation_and_speed_profile_are_metric_and_monotonic():
    route = np.asarray([[2.0, 0.0], [5.0, 0.0], [5.0, 5.0]])
    positions, yaws = interpolate_route_by_distance(route, np.asarray([1.0, 5.0, 8.0]))
    np.testing.assert_allclose(positions, [[1, 0], [5, 0], [5, 3]], atol=1e-6)
    np.testing.assert_allclose(yaws, [0, np.pi / 2, np.pi / 2], atol=1e-6)
    distances = speed_profile_distances(0.0, 8.0)
    assert np.all(np.diff(distances) >= 0)
    assert distances[-1] <= 4.0
