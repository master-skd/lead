"""Scalar-speed actions and controller-in-loop longitudinal rollout.

The throttle/brake decision calls the same controller as closed-loop inference.
CARLA vehicle dynamics are approximated explicitly; this is an offline oracle,
not a replay of CARLA or of receding-horizon model decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lead.common.pid_controller import get_throttle
from lead.data_loader.future_actor_cache import FUTURE_TIMES_S
from lead.expert.config_expert import ExpertConfig

ACTION_NAMES = ("baseline", "slow_0p5", "slow_1", "slow_2", "stop", "faster_1")


@dataclass(frozen=True)
class Dynamics:
    drive_accel_mps2: float = 4.8
    brake_decel_mps2: float = 6.0
    drag_mps2: float = 0.2
    dt_s: float = 0.05
    brake_ratio: float = 1.1


@dataclass(frozen=True)
class ActionRollout:
    target_mps: float
    xy_distance_m: np.ndarray  # [8] along the frozen Path at 4 Hz
    speed_mps: np.ndarray  # [8]
    acceleration_mps2: np.ndarray  # [40]
    brake_ticks: int
    throttle_ticks: int
    comfort: float

    @property
    def progress_m(self) -> float:
        return float(self.xy_distance_m[-1])


def scalar_actions(
    raw_target_mps: float, *, max_target_mps: float = 15.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return baseline and five interpretable counterfactual scalar commands.

    Repeated commands (common when the baseline asks for a stop) are masked,
    while index zero is always the exact, unmodified baseline target.
    """
    raw = max(0.0, float(raw_target_mps))
    targets = np.array(
        [
            raw,
            max(0.0, raw - 0.5),
            max(0.0, raw - 1.0),
            max(0.0, raw - 2.0),
            0.0,
            min(raw + 1.0, max(raw, max_target_mps)),
        ],
        dtype=np.float32,
    )
    valid = np.ones(len(targets), dtype=bool)
    for index in range(1, len(targets)):
        valid[index] = not np.any(
            np.isclose(targets[index], targets[:index], atol=0.01)
        )
    return targets, valid


def rollout_scalar_target(
    current_speed_mps: float,
    target_speed_mps: float,
    *,
    dynamics: Dynamics | None = None,
    expert_config: ExpertConfig | None = None,
    times_s: np.ndarray = FUTURE_TIMES_S,
) -> ActionRollout:
    """Hold one scalar target for 2 s through LEAD's longitudinal controller.

    Only the vehicle acceleration model is approximate. The target-to-command
    mapping, including the abrupt full-brake threshold, is the production code.
    """
    if expert_config is None:
        expert_config = ExpertConfig()
    if dynamics is None:
        dynamics = Dynamics()
    times = np.asarray(times_s, dtype=np.float32)
    if len(times) == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times_s must be a nonempty increasing array")
    interval_ticks = np.rint(np.diff(np.r_[0.0, times]) / dynamics.dt_s).astype(int)
    if np.any(interval_ticks < 1) or not np.allclose(
        np.cumsum(interval_ticks) * dynamics.dt_s, times, atol=1e-4
    ):
        raise ValueError("times_s must align with dynamics.dt_s")
    current = max(0.0, float(current_speed_mps))
    target = max(0.0, float(target_speed_mps))
    distance = 0.0
    sampled_distance = []
    sampled_speed = []
    accelerations = []
    brake_ticks = 0
    throttle_ticks = 0
    for count in interval_ticks:
        for _ in range(int(count)):
            brake_requested = target < 0.01 or (
                target > 0.0 and current / target > dynamics.brake_ratio
            )
            throttle, brake = get_throttle(
                bool(brake_requested), target, current, expert_config
            )
            acceleration = (
                dynamics.drive_accel_mps2 * throttle
                - dynamics.brake_decel_mps2 * float(brake)
                - dynamics.drag_mps2 * float(current > 0.0)
            )
            next_speed = max(0.0, current + acceleration * dynamics.dt_s)
            distance += 0.5 * (current + next_speed) * dynamics.dt_s
            current = next_speed
            accelerations.append(acceleration)
            brake_ticks += bool(brake)
            throttle_ticks += throttle > 0.01
        sampled_distance.append(distance)
        sampled_speed.append(current)
    acceleration_array = np.asarray(accelerations, dtype=np.float32)
    jerk = np.diff(acceleration_array) / dynamics.dt_s
    # Interpretable, bounded comfort proxy, not a learned score or ground truth.
    comfort = float(
        np.exp(
            -0.08 * np.mean(np.abs(acceleration_array))
            - 0.002 * (np.mean(np.abs(jerk)) if len(jerk) else 0.0)
        )
    )
    return ActionRollout(
        target_mps=target,
        xy_distance_m=np.asarray(sampled_distance, dtype=np.float32),
        speed_mps=np.asarray(sampled_speed, dtype=np.float32),
        acceleration_mps2=acceleration_array,
        brake_ticks=brake_ticks,
        throttle_ticks=throttle_ticks,
        comfort=comfort,
    )
