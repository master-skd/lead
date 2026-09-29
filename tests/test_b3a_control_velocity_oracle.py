import numpy as np

from scripts.p4.audit_b3a_control_velocity_oracle import (
    control_target,
    hold_target_distances,
    interpolate_samples,
    reconstruct_route_samples,
)


def test_control_target_matches_pid_mapping_and_clips_to_zero():
    assert control_target(2.0, 2.5) == 3.0
    assert control_target(4.0, 1.0) == 0.0
    np.testing.assert_allclose(
        hold_target_distances(2.5, 3.0)[:2], [0.625, 1.375]
    )


def test_reconstructed_fixed_route_recovers_cached_positions():
    velocity = np.array([
        [2.0] * 8,
        [4.0] * 8,
    ], dtype=np.float32)
    states = np.zeros((2, 8, 6), dtype=np.float32)
    distances = np.cumsum(velocity, axis=1) * 0.25
    states[..., 0] = distances
    states[..., 3] = 1.0
    arc = reconstruct_route_samples(velocity, states, np.array([True, True]))
    xy, yaw = interpolate_samples(arc, np.array([0.25, 0.5, 2.0, 8.0]))
    np.testing.assert_allclose(xy[:, 0], [0.25, 0.5, 2.0, 8.0])
    np.testing.assert_allclose(xy[:, 1], 0.0)
    np.testing.assert_allclose(yaw, 0.0)
