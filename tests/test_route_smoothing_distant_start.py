import numpy as np

from lead.data_loader.carla_dataset_utils import (
    iterative_line_interpolation,
    smooth_path,
)
from lead.training.config_training import TrainingConfig


def test_distant_first_waypoint_resamples_from_ego():
    config = TrainingConfig()
    config.num_route_points_smoothing = 40
    # A 39-point route starting 35 m ahead used to exhaust the legacy iterator.
    x = 35.415 + np.arange(39, dtype=np.float64) * 0.9563
    route = np.column_stack((x, -0.023 - np.arange(39) * 0.2925))

    result = smooth_path(config, route, target_first_distance=2.5)

    assert result.shape == (40, 2)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(np.linalg.norm(result[0]), 2.5, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(np.diff(result, axis=0), axis=1), 1.0, atol=0.02)
    assert result[-1, 0] < route[-1, 0]


def test_near_first_waypoint_preserves_legacy_interpolation():
    config = TrainingConfig()
    config.num_route_points_smoothing = 40
    route = np.column_stack((np.arange(2.5, 52.5), np.zeros(50)))

    actual = smooth_path(config, route, target_first_distance=2.5)
    legacy = iterative_line_interpolation(config, route, target_first_distance=2.5)

    np.testing.assert_array_equal(actual, legacy)
