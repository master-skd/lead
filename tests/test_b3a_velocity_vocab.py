import numpy as np

from scripts.p4.build_b3a_velocity_vocab import (
    interval_average_profile,
    parse_clusters,
    sort_centers,
    summarize_quantization,
)


def test_interval_average_profile_matches_linear_speed_integral():
    times = np.arange(41, dtype=np.float32) * 0.05
    profile = interval_average_profile(times, steps=8, interval_s=0.25, raw_dt_s=0.05)
    np.testing.assert_allclose(profile, np.arange(8) * 0.25 + 0.125, atol=1e-6)


def test_interval_average_profile_clamps_tiny_negative_speeds():
    profile = interval_average_profile(
        np.full(41, -1e-4, dtype=np.float32),
        steps=8,
        interval_s=0.25,
        raw_dt_s=0.05,
    )
    np.testing.assert_array_equal(profile, np.zeros(8, dtype=np.float32))


def test_quantization_reports_exact_two_mode_coverage():
    centers = np.stack((np.zeros(8), np.full(8, 4.0))).astype(np.float32)
    arrays = {
        "profiles": centers.copy(),
        "current_speed": np.array([0.0, 4.0], dtype=np.float32),
    }
    summary, assignment = summarize_quantization(arrays, centers, 0.25)
    np.testing.assert_array_equal(assignment, np.array([0, 1]))
    assert summary["nearest_profile_mae_mps"]["mean"] == 0.0
    assert summary["two_second_progress_abs_error_m"]["mean"] == 0.0
    assert summary["coverage"]["mae_le_0.25_mps"] == 1.0


def test_center_sort_is_slow_to_fast_and_cluster_parser_is_stable():
    centers = np.stack((np.full(8, 5.0), np.full(8, 1.0), np.full(8, 3.0)))
    sorted_centers = sort_centers(centers)
    np.testing.assert_allclose(sorted_centers.mean(axis=1), [1.0, 3.0, 5.0])
    assert parse_clusters("64,16,32,32") == (16, 32, 64)
