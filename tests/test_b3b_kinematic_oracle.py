import numpy as np
import pytest

from scripts.p4.eval_b3b_kinematic_oracle import prepared_route, rollout, route_point


def test_straight_rollout_integrates_speed_and_heading_consistently():
    route = np.stack((np.arange(1, 11), np.zeros(10)), axis=-1)
    positions, yaws, error = rollout(route, np.full(8, 2.0))
    np.testing.assert_allclose(positions[:, 0], np.arange(1, 9) * 0.5, atol=1e-6)
    np.testing.assert_allclose(positions[:, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(yaws, 0.0, atol=1e-6)
    assert error == pytest.approx(0.0, abs=1e-6)


def test_turning_rollout_caps_yaw_and_integrates_the_same_pose():
    route = np.array([[1.0, 0.5], [2.0, 2.0], [2.0, 4.0], [2.0, 6.0]])
    speeds = np.full(8, 2.0)
    positions, yaws, _ = rollout(route, speeds, max_yaw_rate_deg_s=30.0)
    assert np.all(np.diff(np.concatenate(([0.0], yaws))) <= np.deg2rad(7.5) + 1e-6)
    steps = np.diff(np.concatenate((np.zeros((1, 2)), positions)), axis=0)
    np.testing.assert_allclose(np.linalg.norm(steps, axis=1), speeds * 0.25, atol=1e-6)
    assert positions[-1, 1] > 0


def test_stopped_ego_does_not_rotate_in_place():
    route = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    positions, yaws, _ = rollout(route, np.zeros(8))
    np.testing.assert_allclose(positions, 0.0)
    np.testing.assert_allclose(yaws, 0.0)


def test_target_extrapolates_past_route_endpoint():
    prepared = prepared_route(np.array([[1.0, 0.0], [2.0, 0.0]]))
    np.testing.assert_allclose(route_point(prepared, 4.0), [4.0, 0.0])
