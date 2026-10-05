import socket
import struct
import threading
from types import SimpleNamespace

import numpy as np
import torch

from lead.inference.vlm_client import VLMServiceClient
from lead.tfv6.joint_online_training import build_same_pass_labels, compose_route_states
from lead.tfv6.joint_trajectory_scorer import JointTrajectorySceneScorer, joint_score_loss
from lead.tfv6.planning_decoder import PlanningDecoder


def test_joint_states_and_labels_for_two_live_arms():
    routes = torch.stack((
        torch.stack((torch.arange(1, 31).float(), torch.zeros(30)), dim=-1),
        torch.stack((torch.arange(1, 31).float(), torch.full((30,), 8.0)), dim=-1),
    )).unsqueeze(0)
    states = compose_route_states(routes, torch.tensor([3.0]), torch.tensor([4.0]))
    assert states.shape == (1, 2, 8, 6)
    assert not states.requires_grad

    corridor = torch.zeros(1, 1, 320, 384)
    corridor[:, :, 155:165, :] = 1
    data = {
        "joint_actor_counts": torch.tensor([0]),
        "joint_actor_positions": torch.zeros(1, 90, 8, 2),
        "joint_actor_yaws": torch.zeros(1, 90, 8),
        "joint_actor_extents": torch.zeros(1, 90, 3),
        "joint_actor_z": torch.zeros(1, 90),
        "joint_actor_valid": torch.zeros(1, 90, 8, dtype=torch.bool),
        "joint_actor_class_ids": torch.zeros(1, 90, dtype=torch.uint8),
        "joint_actor_ego_extent": torch.tensor([[2.45, 0.95, 0.75]]),
        "joint_ego_future": states[:, 0, :, :2].clone(),
        "route": routes[:, 0],
        "visual_intent_label": corridor,
    }
    config = SimpleNamespace(pixels_per_meter=4.0, min_x_meter=-16.0, min_y_meter=-40.0)
    labels = build_same_pass_labels(states, torch.ones(1, 2, dtype=torch.bool), data, config)
    assert set(labels) == {"collision_free", "ttc", "drivable", "task", "progress", "comfort", "imitation"}
    assert labels["collision_free"].tolist() == [[1.0, 1.0]]
    assert labels["imitation"][0, 0] == 1
    assert labels["imitation"][0, 1] < 1
    # A GT future actor occupying the first route must flip its counterfactual
    # collision target without changing the second route's label.
    data["joint_actor_counts"][0] = 1
    data["joint_actor_positions"][0, 0] = states[0, 0, :, :2]
    data["joint_actor_extents"][0, 0] = torch.tensor([1.0, 0.5, 0.5])
    data["joint_actor_valid"][0, 0] = True
    data["joint_actor_class_ids"][0, 0] = 1
    collision_labels = build_same_pass_labels(
        states, torch.ones(1, 2, dtype=torch.bool), data, config,
    )
    assert collision_labels["collision_free"][0, 0] == 0


def test_vlm_batch_wire_protocol_preserves_fp16():
    client_socket, server_socket = socket.socketpair()
    client = object.__new__(VLMServiceClient)
    client.sock = client_socket
    images = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
    expected = np.arange(2 * 2 * 3 * 4, dtype=np.float16).reshape(2, 2, 3, 4)

    def server():
        header = server_socket.recv(16)
        assert struct.unpack("<IIII", header) == (0, 2, 3, 4)
        payload = bytearray()
        while len(payload) < images.nbytes:
            payload.extend(server_socket.recv(images.nbytes - len(payload)))
        np.testing.assert_array_equal(np.frombuffer(payload, dtype=np.uint8).reshape(images.shape), images)
        server_socket.sendall(struct.pack("<IIII", 2, 2, 3, 4) + expected.tobytes())
        server_socket.close()

    worker = threading.Thread(target=server)
    worker.start()
    actual = client.extract_batch(images)
    np.testing.assert_array_equal(actual, expected)
    client.close()
    worker.join()


def test_scorer_gradient_does_not_reach_planner_routes_or_scene():
    routes = torch.randn(1, 2, 30, 2, requires_grad=True)
    scene = torch.randn(1, 8, 256, requires_grad=True)
    states = compose_route_states(routes, torch.tensor([2.0]), torch.tensor([3.0]))
    scorer = JointTrajectorySceneScorer(
        hidden_dim=32, num_heads=4, temporal_layers=1, interaction_layers=1,
    )
    valid = torch.tensor([[True, True]])
    logits = scorer(states.detach(), scene.detach(), valid)
    targets = {name: torch.ones(1, 2) for name in logits}
    loss, _ = joint_score_loss(logits, targets, valid)
    loss.backward()
    assert routes.grad is None
    assert scene.grad is None
    assert scorer.heads["task"][-1].weight.grad is not None


def test_multimodal_planner_keeps_geometry_loss_without_confidence_loss():
    decoder = object.__new__(PlanningDecoder)
    torch.nn.Module.__init__(decoder)
    decoder.device = torch.device("cpu")
    decoder.config = SimpleNamespace(
        joint_scorer_replaces_conf=True, route_near_points=1, route_far_weight=0.3,
        route_anchor_tol_deg=25.0, route_anchor_min_reach_frac=0.5,
        use_collision_cost=False, use_route_corridor_loss=False,
        route_pad_conf_loss_weight=0.0,
    )
    routes = torch.tensor([[[[1., 0.], [2., 0.]], [[1., 1.], [2., 1.]]]], requires_grad=True)
    data = {
        "route_multimodal": routes,
        "anchor": torch.tensor([[[0., 1., 0.1, 1.], [0.7, 0.7, 0.1, 1.]]]),
    }
    loss = {}
    decoder._multimodal_route_loss(
        torch.tensor([[[1., 0.], [2., 0.]]]), data, loss, {},
    )
    assert "loss_spatial_route" in loss
    assert "loss_route_anchor" in loss
    assert "loss_route_conf" not in loss
    assert "loss_route_pad_conf" not in loss
