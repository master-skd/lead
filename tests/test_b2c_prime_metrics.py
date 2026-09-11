import torch

from scripts.p4.eval_b2c_prime_speed_gate import _scope_metrics


def test_expert_consistency_metrics_only_count_incremental_interventions():
    result = _scope_metrics(
        pred_risk=torch.tensor([0.8, 0.8, 0.2, 0.2]),
        gt_risk=torch.tensor([0.8, 0.0, 0.8, 0.0]),
        raw_speed=torch.tensor([8.0, 8.0, 8.0, 0.0]),
        expert_speed=torch.tensor([0.0, 8.0, 8.0, 0.0]),
        expert_brake=torch.tensor([True, False, False, True]),
        route_ade=torch.zeros(4),
        pairs=[(0.5, 0.9)],
        gt_threshold=0.1,
    )
    common = result["common"]
    gate = result["gates"]["0.5,0.9"]

    assert common["expert_brake_rate"] == 0.5
    assert common["model_brake_rate"] == 0.25
    # One expert-brake frame was already stopped; only the moving miss is recoverable.
    assert common["missed_expert_brake_rate"] == 0.5
    assert gate["missed_expert_brake_slow_recovery"] == 1.0
    assert gate["missed_expert_brake_stop_recovery"] == 0.0
    # Of two moving expert-go frames, only the high predicted-risk one is slowed.
    assert gate["false_slow_rate_on_expert_go"] == 0.5
    assert gate["false_stop_rate_on_expert_go"] == 0.0
    assert gate["unsafe_recall"] == 0.5
    assert gate["slow_precision"] == 0.5
