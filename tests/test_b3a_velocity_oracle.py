import numpy as np

from scripts.p4.eval_b3a_velocity_oracle import (
    build_target_speed_candidates,
    select_minimal_slowdown,
)


def test_candidates_keep_raw_and_only_add_slowdowns():
    candidates, raw_index = build_target_speed_candidates(
        10.0,
        [0.0, 4.0, 8.0, 10.0, 13.9],
        (0.0, 0.25, 0.5, 0.75, 1.0),
    )
    np.testing.assert_allclose(candidates, [0.0, 2.5, 4.0, 5.0, 7.5, 8.0, 10.0])
    assert candidates[raw_index] == 10.0
    assert candidates.max() == 10.0


def test_oracle_keeps_safe_raw_profile():
    speeds = np.asarray([0.0, 4.0, 8.0])
    collision = np.asarray([False, False, False])
    assert select_minimal_slowdown(speeds, collision, raw_index=2) == 2


def test_oracle_uses_fastest_safe_slowdown_and_falls_back_to_raw():
    speeds = np.asarray([0.0, 4.0, 8.0])
    assert (
        select_minimal_slowdown(
            speeds,
            np.asarray([False, False, True]),
            raw_index=2,
        )
        == 1
    )
    assert (
        select_minimal_slowdown(
            speeds,
            np.asarray([True, True, True]),
            raw_index=2,
        )
        == 2
    )
