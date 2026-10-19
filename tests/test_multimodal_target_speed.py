"""CPU regressions for route-conditioned speeds and paired WTA supervision."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lead.tfv6.joint_online_training import compose_route_states, select_route_speed_pair
from lead.tfv6.planning_decoder import PlanningDecoder


def bare_decoder():
    decoder = object.__new__(PlanningDecoder)
    nn.Module.__init__(decoder)
    decoder.device = torch.device("cpu")
    decoder.config = SimpleNamespace(
        predict_spatial_path=False, predict_target_speed=True,
        predict_temporal_spatial_waypoints=False, multimodal_target_speed=True,
        target_speed_classes=[0.0, 2.0, 4.0], torch_float_type=torch.float32,
        use_collision_cost=False,
    )
    return decoder


def test_paired_speed_states_and_legacy_broadcast():
    routes = torch.stack((torch.arange(1, 31).float(), torch.zeros(30)), -1)
    routes = routes[None, None].expand(2, 3, -1, -1).clone().requires_grad_()
    speeds = torch.tensor([[1., 3., 5.], [5., 3., 1.]], requires_grad=True)
    states = compose_route_states(routes, torch.tensor([2., 2.]), speeds)
    assert states.shape == (2, 3, 8, 6)
    torch.testing.assert_close(states[:, :, -1, 4], speeds.detach())
    assert states[0, 0, -1, 0] < states[0, 2, -1, 0]
    assert not states.requires_grad
    shared = compose_route_states(routes, torch.tensor([2., 2.]), torch.tensor([3., 4.]))
    paired = compose_route_states(routes, torch.tensor([2., 2.]), torch.tensor([[3., 3., 3.], [4., 4., 4.]]))
    torch.testing.assert_close(shared, paired)
    with pytest.raises(ValueError, match="target_speed"):
        compose_route_states(routes, torch.tensor([2., 2.]), torch.ones(2, 1))


def test_inference_gathers_route_and_speed_from_same_scorer_index():
    routes = torch.arange(24).float().reshape(2, 3, 2, 2)
    logits = torch.randn(2, 3, 4)
    speeds = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    route, distribution, speed = select_route_speed_pair(routes, logits, speeds, torch.tensor([2, 0]))
    torch.testing.assert_close(route, torch.stack((routes[0, 2], routes[1, 0])))
    torch.testing.assert_close(distribution, torch.stack((logits[0, 2], logits[1, 0])))
    torch.testing.assert_close(speed, torch.tensor([3., 4.]))


def test_speed_loss_only_supervises_same_valid_route_winner():
    decoder = bare_decoder()
    routes = torch.tensor([[[[1., 4.], [2., 4.]],
                            [[1., 1.], [2., 1.]],
                            [[1., 0.], [2., 0.]]]], requires_grad=True)
    logits = torch.randn(1, 3, 3, requires_grad=True)
    data = {
        "route_multimodal": routes,
        "route": routes[:, 2].detach(),
        "anchor": torch.tensor([[[0., 1., 1., 1.], [0., 1., 1., 1.], [0., 1., 1., 0.]]]),
        "target_speed_multimodal_distribution": logits,
        "target_speed": torch.tensor([3.]), "brake": torch.tensor([False]),
    }
    assert decoder._route_winner(data["route"], data).tolist() == [1]
    loss = {}
    decoder.compute_loss(SimpleNamespace(pred_target_speed_distribution=logits[:, 0]), data, loss, {})
    loss["loss_target_speed"].backward()
    assert logits.grad[0, 1].abs().sum() > 0
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[0, 2].abs().sum() == 0
    assert routes.grad is None  # WTA assignment is discrete, not scorer feedback


class Context(nn.Module):
    def forward(self, **kwargs):
        return torch.zeros(1, 2, 4)


class PassQueries(nn.Module):
    def forward(self, queries, context):
        return queries


def test_forward_emits_k_route_conditioned_speed_distributions():
    decoder = bare_decoder()
    decoder.mm = decoder.mm_speed = True
    decoder.K = 2
    decoder.config.predict_spatial_path = True
    decoder.config.num_route_points_prediction = 2
    decoder.config.predict_temporal_spatial_waypoints = True
    decoder.config.num_way_points_prediction = 2
    decoder.config.use_navsim_data = False
    decoder.config.joint_scorer_replaces_conf = True
    decoder.config.route_select_by_conf = False
    decoder.planning_context_encoder = Context()
    decoder.transformer_decoder = PassQueries()
    decoder.query = nn.Parameter(torch.zeros(1, 7, 4))
    decoder.anchor_embed = nn.Linear(4, 4, bias=False)
    decoder.route_decoder = nn.Linear(4, 2)
    decoder.wp_decoder = nn.Linear(4, 2)
    decoder.target_speed_decoder = nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        decoder.anchor_embed.weight.copy_(torch.eye(4))
        decoder.target_speed_decoder.weight.copy_(torch.eye(4)[:3])
    data = {}
    decoder(torch.zeros(1, 4, 2, 2), None, None, data, {},
            anchor=torch.tensor([[[0., 1., 1., 1.], [1., 0., 1., 1.]]]))
    assert data["route_multimodal"].shape == (1, 2, 2, 2)
    logits = data["target_speed_multimodal_distribution"]
    assert logits.shape == (1, 2, 3)
    assert not torch.equal(logits[:, 0], logits[:, 1])
    assert data["target_speed_multimodal_scalar"].shape == (1, 2)
    logits[:, 1].sum().backward()
    assert decoder.anchor_embed.weight.grad is not None
