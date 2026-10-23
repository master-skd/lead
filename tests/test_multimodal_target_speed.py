"""CPU regressions for route-conditioned speeds and paired WTA supervision."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lead.tfv6.joint_online_training import compose_route_states, select_route_speed_pair
from lead.tfv6.planning_decoder import PlanningDecoder
from lead.tfv6.tfv6 import TFv6


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


def test_joint_scorer_choice_survives_legacy_route_selector(monkeypatch):
    """Closed loop must keep the scorer's route and its paired target speed."""
    routes = torch.zeros(1, 2, 10, 2)
    routes[0, 1, :, 1] = 5.0
    speeds = torch.tensor([[2.0, 7.0]])
    distributions = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    class Backbone(nn.Module):
        def forward(self, data):
            return torch.zeros(1, 4, 1, 1), None

        def top_down(self, features):
            return features

    class Planner(nn.Module):
        kv = torch.zeros(1, 1, 4)

        def forward(self, bev, radar_features, radar_predictions, data, log, **kwargs):
            data["route_multimodal"] = routes
            data["route_conf"] = torch.tensor([[0.9, 0.1]])
            data["route_features"] = torch.zeros(1, 2, 4)
            data["target_speed_multimodal_distribution"] = distributions
            data["target_speed_multimodal_scalar"] = speeds
            return routes[:, 0], None, distributions[:, 0], speeds[:, 0], None

    class Scorer(nn.Module):
        def forward(self, states, scene, valid):
            names = (
                "collision_free", "ttc", "drivable", "task", "progress",
                "comfort", "imitation",
            )
            return {name: torch.tensor([[0.0, 10.0]]) for name in names}

    monkeypatch.setattr(
        "lead.tfv6.trajectory_scene_scorer.compress_planner_scene_tokens",
        lambda *args, **kwargs: torch.zeros(1, 1, 4),
    )
    model = object.__new__(TFv6)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.config = SimpleNamespace(
        use_carla_data=False, radar_detection=False, use_radar_detection=False,
        use_vlm_intent=False,
        use_intent_decoder=False, use_planning_decoder=True,
        multimodal_planner=True, use_control_conditioning=False,
        online_joint_training=True, joint_scorer_replaces_conf=True,
        multimodal_target_speed=True, joint_scorer_steps=4,
        lidar_vert_anchors=1, lidar_horz_anchors=1, use_semantic=False,
        use_depth=False, detect_boxes=False, use_bev_semantic=False,
        use_navsim_data=False, route_speed_safety_gate=False,
        route_selection_mode="legacy", route_select_by_conf=False,
    )
    model.backbone = Backbone()
    model.planning_decoder = Planner()
    model.joint_scorer = Scorer()
    model.eval()
    prediction = model({
        "anchor": torch.tensor([[[0., 1., 1., 1.], [0., 1., 1., 1.]]]),
        "speed": torch.tensor([3.0]),
    })

    assert prediction.pred_route_selected_idx.tolist() == [1]
    torch.testing.assert_close(prediction.pred_route, routes[:, 1])
    torch.testing.assert_close(prediction.pred_target_speed_scalar, speeds[:, 1])
    torch.testing.assert_close(
        prediction.pred_target_speed_distribution, distributions[:, 1],
    )
    torch.testing.assert_close(prediction.pred_route_multimodal, routes)
