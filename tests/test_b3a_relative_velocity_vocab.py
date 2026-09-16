import numpy as np

from scripts.p4.build_b3a_relative_velocity_vocab import (
    nearest_relative_profiles,
    realize_relative_profiles,
    sort_relative_centers,
)


def test_relative_profiles_are_shifted_by_current_speed_and_clamped():
    centers = np.asarray([[0.0, -1.0, -3.0], [0.5, 1.0, 1.5]], dtype=np.float32)
    realized = realize_relative_profiles(centers, np.asarray([2.0, 0.5]))
    np.testing.assert_allclose(realized[0], [[2.0, 1.0, 0.0], [2.5, 3.0, 3.5]])
    np.testing.assert_allclose(realized[1], [[0.5, 0.0, 0.0], [1.0, 1.5, 2.0]])


def test_relative_nearest_neighbor_is_conditioned_on_current_speed():
    centers = np.asarray([[0.0, 0.0], [-1.0, -2.0]], dtype=np.float32)
    current = np.asarray([5.0, 1.0], dtype=np.float32)
    expert = np.asarray([[5.0, 5.0], [0.0, 0.0]], dtype=np.float32)
    assignment, error = nearest_relative_profiles(expert, current, centers)
    np.testing.assert_array_equal(assignment, [0, 1])
    np.testing.assert_allclose(error, 0.0)


def test_relative_centers_sort_from_braking_to_accelerating():
    centers = np.asarray([[1.0, 2.0], [-1.0, -2.0], [0.0, 0.0]])
    sorted_centers = sort_relative_centers(centers)
    np.testing.assert_allclose(sorted_centers, [[-1.0, -2.0], [0.0, 0.0], [1.0, 2.0]])
