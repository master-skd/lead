"""Live P7 bridge tests without a CARLA server or Qwen checkpoint."""
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lead.inference.p7_closed_loop_intent import P7ClosedLoopIntent
from lead.inference.qwendrive_intent_client import QwenDriveIntentClient
from lead.inference.qwendrive_intent_service import serve
from lead.inference.vlm_image_preprocess import carla_bgr_to_vlm_rgb
from lead.tfv6.semantic_intent_slots import service_results_to_tensors


def proposal():
    return {'scene_rules': {'signal': 'red', 'stop_sign': 'no',
                           'pedestrian_conflict': 'no', 'vehicle_conflict': 'uncertain'},
            'intents': [{'maneuver': 'keep_lane', 'obligations': ['stop_for_red'],
                         'confidence': 0.8, 'evidence': 'red signal'}]}


def test_live_pixels_prompts_and_semantics_match_training():
    raw = np.zeros((384, 1152, 3), dtype=np.uint8)
    raw[:, :384] = [20, 30, 40]
    raw[:, 384:768] = [100, 110, 120]
    raw[:, 768:] = [200, 210, 220]
    expected = carla_bgr_to_vlm_rgb(raw, 30)
    calls = []
    class Model:
        def generate_text(self, images, question, **kwargs):
            calls.append((images, question, kwargs))
            return SimpleNamespace(text='observed signal' if len(calls) == 1 else json.dumps(proposal()))
    client_socket, server_socket = socket.socketpair()
    client = QwenDriveIntentClient.__new__(QwenDriveIntentClient)
    client.sock, client.requests = client_socket, 0
    thread = Thread(target=serve, args=(Model(), server_socket, None),
                    kwargs={'analysis_tokens': 512, 'answer_tokens': 384})
    thread.start()
    bridge = P7ClosedLoopIntent(client)
    try:
        live = bridge.prepare(raw, [0, 0, 0, 1, 0, 0], 3.0)
    finally:
        bridge.close()
        thread.join(timeout=5)
        server_socket.close()
    assert not thread.is_alive()
    assert client.requests == 1
    assert len(calls) == 2
    for images, prompt, limits in calls:
        for image, expected_view in zip(images, [expected[:, 384:768], expected[:, :384], expected[:, 768:]], strict=True):
            np.testing.assert_array_equal(np.asarray(image), expected_view)
        assert 'Navigation: follow_lane' in prompt and '3.0 m/s' in prompt
    assert calls[0][2]['max_new_tokens'] == 512
    assert calls[1][2]['max_new_tokens'] == 384
    assert 'observed signal' in calls[1][1]
    offline = service_results_to_tensors([{'raw_text': json.dumps(proposal()), 'error': None}],
                                        ['follow_lane'], capacity=6)
    assert all(torch.equal(live[key], value) for key, value in offline.items())
    assert bridge.last_trace['real_valid_branches'] == 1
    assert bridge.last_trace['fallback'] is False


def test_live_bad_json_falls_back_but_connection_failures_abort():
    class Client:
        def generate_rgb(self, *_): return [{'raw_text': 'broken', 'error': None}]
        def close(self): pass
    bridge = P7ClosedLoopIntent(Client())
    image = np.zeros((384, 1152, 3), dtype=np.uint8)
    result = bridge.prepare(image, [0, 0, 0, 1, 0, 0], 0.0)
    assert result['semantic_intent_fallback'].tolist() == [True]
    assert result['semantic_intent_valid'].sum() == 1
    assert bridge.last_trace['real_valid_branches'] == 0
    def fail(*_): raise ConnectionError('service died')
    bridge.client.generate_rgb = fail
    with pytest.raises(ConnectionError, match='service died'):
        bridge.prepare(image, [0, 0, 0, 1, 0, 0], 0.0)


def test_p7_eval_preflight_exposes_exactly_one_checkpoint(tmp_path):
    project = tmp_path / 'project'
    scripts = project / 'scripts/p7'
    scripts.mkdir(parents=True)
    source = Path('scripts/p7/eval_p7_b2d.sh')
    shutil.copyfile(source, scripts / source.name)
    (project / 'data/benchmark_routes/bench2drive').mkdir(parents=True)
    carla = project / 'carla'
    carla.mkdir()
    (carla / 'CarlaUE4.sh').write_text('#!/bin/sh\nexit 0\n')
    (carla / 'CarlaUE4.sh').chmod(0o755)
    checkpoint = project / 'training'
    checkpoint.mkdir()
    for name in ('model_0018.pth', 'model_0019.pth'):
        (checkpoint / name).touch()
    config = {'use_semantic_intent_planner': True, 'multimodal_planner': True,
              'multimodal_target_speed': True, 'joint_scorer_replaces_conf': False}
    (checkpoint / 'config.json').write_text(json.dumps(config))
    env = {**os.environ, 'P7_B2D_LEAD_PYTHON': sys.executable,
           'P7_B2D_QWEN_PYTHON': sys.executable, 'P7_B2D_QWEN_MODEL': str(project),
           'CARLA_ROOT': str(carla), 'P7_B2D_PREFLIGHT_ONLY': '1'}
    args = ['bash', str(scripts / source.name), str(checkpoint / 'model_0019.pth'), '0 1', 'p7_test']
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    exposed = project / 'outputs/p7_b2d_eval_checkpoints/p7_test'
    assert sorted(path.name for path in exposed.iterdir()) == ['config.json', 'model.pth']
    assert (exposed / 'model.pth').resolve() == checkpoint / 'model_0019.pth'
    config['joint_scorer_replaces_conf'] = True
    (checkpoint / 'config.json').write_text(json.dumps(config))
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and 'requires a P7' in result.stderr
