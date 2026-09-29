import numpy as np

from scripts.p4.audit_b3b_pid_joint_coverage import eligible_actions, choose_safe


def test_eligibility_requires_task_corridor_and_executable_slowdown():
    eligible = eligible_actions(
        route_valid=np.array([True, True, False]),
        direction_ok=np.array([True, False, True]),
        corridor_cost=np.array([0.05, 0.0, 0.0]),
        route_length=np.array([8.0, 8.0, 8.0]),
        target_valid=np.array([True, True, True, True]),
        targets=np.array([4.0, 3.0, 5.0, 0.0]),
        progress=np.array([6.0, 4.5, 7.0, 0.5]),
        raw_target=4.0,
        max_corridor_cost=0.1,
        max_progress_loss_m=2.0,
    )
    assert eligible.tolist() == [
        [True, True, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]


def test_selection_excludes_collisions_tracking_error_and_raw():
    eligible = np.ones((3, 2), dtype=bool)
    collision = np.array([[True, False], [False, False], [False, False]])
    error = np.array([[0.0, 0.5], [1.5, 0.5], [0.5, 0.5]])
    progress = np.array([6.0, 5.0])
    assert choose_safe(eligible, collision, error, progress, 0, 1.0, 1) == (0, 1)
    error[0, 1] = 1.5
    assert choose_safe(eligible, collision, error, progress, 0, 1.0, 1) is None
    assert choose_safe(eligible, collision, error, progress, 0, 1.0, 2) == (1, 1)
    assert choose_safe(eligible, collision, error, progress, 0, 1.0, 3) == (2, 0)
