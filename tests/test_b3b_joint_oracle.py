import numpy as np
import pytest

from scripts.p4.eval_b3b_joint_oracle import (
    candidate_eligibility,
    choose_oracle,
    expert_direction_mask,
    local_path_variants,
    summarize,
)


def test_joint_oracle_rescues_only_with_another_path():
    collision = np.array([[True, True], [True, False]])
    eligible = np.ones((2, 2), dtype=bool)
    progress = np.array([[10.0, 8.0], [10.0, 7.0]])
    assert choose_oracle(True, collision, eligible, progress, 0, joint=False) == (0, 0)
    assert choose_oracle(True, collision, eligible, progress, 0, joint=True) == (1, 1)
    assert choose_oracle(False, collision, eligible, progress, 0, joint=True) == (0, 0)
    eligible[1, 1] = False
    assert choose_oracle(True, collision, eligible, progress, 0, joint=True) == (0, 0)


def test_direction_proxy_filters_opposite_path():
    routes = np.array([[[[5.0, 0.0]], [[-5.0, 0.0]], [[5.0, 5.0]]]])
    expert = np.array([[[10.0, 0.0]]])
    np.testing.assert_array_equal(
        expert_direction_mask(routes, expert, max_error_deg=50.0),
        [[True, False, True]],
    )


def test_local_path_variants_are_smooth_and_keep_forward_progress():
    route = np.stack((np.arange(2, 22, 2), np.zeros(10)), axis=-1)[None].astype(float)
    assert local_path_variants(route, ()).shape == (1, 0, 10, 2)
    paths = local_path_variants(route, (-1.0, 1.0), ramp_m=8.0)
    assert paths.shape == (1, 2, 10, 2)
    np.testing.assert_allclose(
        paths[0, :, :, 0], np.broadcast_to(route[0, :, 0], (2, 10))
    )
    assert abs(paths[0, 0, 0, 1]) < 0.2
    assert abs(paths[0, 1, 0, 1]) < 0.2
    assert paths[0, 0, -1, 1] == pytest.approx(-1.0)
    assert paths[0, 1, -1, 1] == pytest.approx(1.0)


def test_candidate_eligibility_applies_navigation_corridor_and_pid_limits():
    kwargs = dict(
        route_valid=np.array([True, True, False]),
        direction_ok=np.array([True, False, True]),
        corridor_cost=np.array([0.02, 0.02, 0.02]),
        route_length=np.array([12.0, 12.0, 12.0]),
        candidate_valid=np.array([True, True, True, False]),
        candidate_progress=np.array([8.0, 6.0, 9.0, 7.0]),
        candidate_pid_target=np.array([4.0, 3.0, 5.0, 3.0]),
        raw_progress=8.0,
        raw_target_speed=4.0,
        max_corridor_cost=0.1,
        min_progress_ratio=0.5,
    )
    reachable = candidate_eligibility(**kwargs, conservative=False)
    conservative = candidate_eligibility(**kwargs, conservative=True)
    np.testing.assert_array_equal(reachable[0], [True, True, True, False])
    np.testing.assert_array_equal(conservative[0], [True, True, False, False])
    assert not reachable[1:].any()


def test_summary_counts_incremental_rescue_only_on_raw_unsafe():
    result = summarize(
        raw_collision=np.array([True, True, False]),
        fixed_rescue=np.array([True, False, False]),
        joint_rescue=np.array([True, True, False]),
        joint_arm=np.array([0, 1, 0]),
        winner_arm=np.zeros(3, dtype=int),
        raw_progress=np.ones(3) * 10,
        joint_progress=np.array([8, 6, 10]),
        scope=np.ones(3, dtype=bool),
    )
    assert result["raw_collision_count"] == 2
    assert result["fixed_path_rescue_count"] == 1
    assert result["joint_rescue_count"] == 2
    assert result["incremental_joint_rescue_count"] == 1
    assert result["changed_path_on_rescue_count"] == 1
    assert result["rescued_progress_ratio_median"] == pytest.approx(0.7)
