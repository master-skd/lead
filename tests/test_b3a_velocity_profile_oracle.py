import numpy as np

from lead.data_loader.future_actor_cache import FUTURE_TIMES_S
from lead.tfv6.future_collision import speed_profile_distances
from scripts.p4.eval_b3a_velocity_profile_oracle import (
    build_velocity_profiles,
    select_max_progress_safe,
    summarize_margin,
)


def _profiles(current_speed=8.0, raw_target=0.0):
    return build_velocity_profiles(
        current_speed,
        raw_target,
        [0.0, 4.0, 8.0],
        (0.0, 0.25, 0.5, 0.75, 1.0),
        (0.5, 1.0, 1.5, 2.0),
        FUTURE_TIMES_S,
        speed_profile_distances,
    )


def test_zero_target_keeps_distinct_earlier_braking_profiles():
    profiles = _profiles()
    assert profiles[0].name == "target_raw"
    assert len(profiles) > 1
    assert any(profile.name == "brake_emergency" for profile in profiles)
    assert (
        min(profile.distances[-1] for profile in profiles) < profiles[0].distances[-1]
    )
    assert all(
        profile.distances[-1] <= profiles[0].distances[-1] + 1e-4
        for profile in profiles
    )


def test_stationary_zero_target_deduplicates_identical_profiles():
    profiles = _profiles(current_speed=0.0, raw_target=0.0)
    assert len(profiles) == 1
    np.testing.assert_allclose(profiles[0].velocities, 0.0)
    np.testing.assert_allclose(profiles[0].distances, 0.0)


def test_oracle_selects_maximum_progress_safe_profile():
    collision = np.asarray([True, False, False])
    progress = np.asarray([10.0, 4.0, 7.0])
    assert select_max_progress_safe(collision, progress) == 2
    assert select_max_progress_safe(np.asarray([False, False, False]), progress) == 0
    assert select_max_progress_safe(np.asarray([True, True, True]), progress) == 0


def test_margin_summary_attributes_incremental_rescue_to_braking_profile():
    arrays = {
        "candidate_valid": np.ones((2, 3), dtype=bool),
        "candidate_collision": np.asarray(
            [[[True, True, False]], [[False, False, False]]]
        ),
        "selected_index": np.asarray([[2], [0]]),
        "candidate_family": np.asarray(
            [["target", "target", "brake"], ["target", "target", "brake"]]
        ),
        "candidate_name": np.asarray(
            [["target_raw", "target_half", "brake_strong"]] * 2
        ),
        "candidate_velocity": np.asarray(
            [
                [[8.0, 8.0], [4.0, 4.0], [2.0, 0.0]],
                [[8.0, 8.0], [4.0, 4.0], [2.0, 0.0]],
            ]
        ),
        "candidate_distance": np.asarray(
            [
                [[1.0, 2.0], [0.5, 1.0], [0.25, 0.5]],
                [[1.0, 2.0], [0.5, 1.0], [0.25, 0.5]],
            ]
        ),
        "candidate_ttc": np.asarray([[[1.0, 1.5, np.inf]], [[np.inf] * 3]]),
        "candidate_overflow": np.zeros((2, 3), dtype=bool),
        "route_ade": np.zeros(2),
        "current_speed": np.full(2, 8.0),
        "raw_target_speed": np.full(2, 8.0),
        "expert_target_speed": np.full(2, 8.0),
        "expert_brake": np.zeros(2, dtype=bool),
    }
    summary = summarize_margin(arrays, 0, np.ones(2, dtype=bool))
    assert summary["raw_collision_rate"] == 0.5
    assert summary["target_only_rescue_count"] == 0
    assert summary["expanded_rescue_count"] == 1
    assert summary["incremental_profile_rescue_count"] == 1
