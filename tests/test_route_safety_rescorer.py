import torch

from lead.tfv6.route_safety_rescorer import (
    resolve_route_selection_mode,
    select_indices_from_costs,
)
from lead.training.config_training import TrainingConfig


def _select(conf, valid, collision, corridor, c_thr=0.1, o_thr=0.1):
    return select_indices_from_costs(
        torch.tensor([conf], dtype=torch.float32),
        torch.tensor([valid], dtype=torch.bool),
        torch.tensor([collision], dtype=torch.float32),
        torch.tensor([corridor], dtype=torch.float32),
        c_thr,
        o_thr,
    )


def test_safe_confidence_winner_is_preserved_not_cost_ranked():
    result = _select([0.9, 0.8], [1, 1], [0.09, 0.01], [0.09, 0.01])
    assert result.selected_idx.item() == 0
    assert not result.switched.item()


def test_unsafe_winner_switches_to_highest_confidence_safe_arm():
    result = _select(
        [0.9, 0.8, 0.7], [1, 1, 1],
        [0.2, 0.01, 0.01], [0.01, 0.01, 0.01],
    )
    assert result.selected_idx.item() == 1
    assert result.switched.item()


def test_off_corridor_gate_rejects_otherwise_collision_safe_arm():
    result = _select(
        [0.9, 0.8, 0.7], [1, 1, 1],
        [0.2, 0.01, 0.02], [0.01, 0.2, 0.02],
    )
    assert result.selected_idx.item() == 2


def test_all_unsafe_falls_back_to_confidence_winner():
    result = _select([0.9, 0.8], [1, 1], [0.2, 0.3], [0.2, 0.3])
    assert result.selected_idx.item() == 0
    assert result.fallback.item()
    assert not result.switched.item()


def test_padding_arm_is_never_selected_even_with_high_confidence():
    result = _select([0.2, 100.0], [1, 0], [0.01, 0.0], [0.01, 0.0])
    assert result.selected_idx.item() == 0
    assert not result.safe_mask[0, 1]


def test_all_padding_has_deterministic_slot_zero_fallback():
    result = _select([0.2, 100.0], [0, 0], [0.0, 0.0], [0.0, 0.0])
    assert result.selected_idx.item() == 0
    assert result.fallback.item()


def test_legacy_mode_resolves_old_boolean():
    config = TrainingConfig()
    config.route_selection_mode = "legacy"
    config.route_select_by_conf = False
    assert resolve_route_selection_mode(config) == "slot0"
    config.route_select_by_conf = True
    assert resolve_route_selection_mode(config) == "confidence"
    config.route_selection_mode = "safety_rescore"
    assert resolve_route_selection_mode(config) == "safety_rescore"
