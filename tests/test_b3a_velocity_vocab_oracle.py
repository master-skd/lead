import numpy as np

from scripts.p4.eval_b3a_velocity_vocab_oracle import (
    interval_speeds_from_distances,
    parse_clusters,
    profile_distances,
    select_progress_oracle,
    start_reachability_mask,
    transition_reachability_mask,
)


def test_profile_distance_round_trip_uses_interval_average_speed():
    profiles = np.asarray([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    distances = profile_distances(profiles, 0.25)
    np.testing.assert_allclose(distances, [[0.25, 0.75, 1.5], [0.0, 0.0, 0.25]])
    np.testing.assert_allclose(
        interval_speeds_from_distances(distances, 0.25), profiles
    )


def test_start_reachability_uses_interval_average_acceleration():
    profiles = np.asarray([[4.0, 4.0], [5.0, 5.0], [2.0, 2.0]], dtype=np.float32)
    reachable = start_reachability_mask(
        profiles, np.asarray([4.0]), 0.25, max_accel_mps2=2.0, max_decel_mps2=4.0
    )
    np.testing.assert_array_equal(reachable, [[True, False, False]])


def test_start_reachability_clamps_negative_carla_speed_noise():
    reachable = start_reachability_mask(
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([-0.08], dtype=np.float32),
        0.25,
    )
    np.testing.assert_array_equal(reachable, [[True]])


def test_transition_reachability_checks_adjacent_interval_means():
    profiles = np.asarray([[1.0, 1.25, 1.5], [1.0, 2.0, 3.0]], dtype=np.float32)
    reachable = transition_reachability_mask(
        profiles, 0.25, max_accel_mps2=2.0, max_decel_mps2=4.0
    )
    np.testing.assert_array_equal(reachable, [True, False])


def test_oracle_keeps_safe_raw_and_selects_max_progress_safe_candidate():
    raw_collision = np.asarray([False, True, True])
    candidate_collision = np.asarray(
        [[False, False, False], [False, False, True], [True, True, True]]
    )
    valid = np.ones((3, 3), dtype=bool)
    selected = select_progress_oracle(
        raw_collision, candidate_collision, valid, np.asarray([1.0, 3.0, 2.0])
    )
    np.testing.assert_array_equal(selected, [0, 2, 0])


def test_oracle_accepts_per_frame_candidate_progress():
    selected = select_progress_oracle(
        np.asarray([True, True]),
        np.zeros((2, 2), dtype=bool),
        np.ones((2, 2), dtype=bool),
        np.asarray([[4.0, 2.0], [1.0, 3.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(selected, [1, 2])


def test_cluster_parser_sorts_and_deduplicates():
    assert parse_clusters("64,16,32,32") == (16, 32, 64)
