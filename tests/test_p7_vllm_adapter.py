"""CPU tests for lossless export and fatal prompt-parity enforcement."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lead.inference.qwendrive_vllm import VLLMQwenDrive, collapse_image_tokens
from lead.inference.qwendrive_intent_client import QwenDriveInputMismatchError


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, f'scripts/p7/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_changes_names_but_preserves_tensor_bytes_and_source(tmp_path):
    script = load_script('prepare_qwendrive_vllm')
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'config.json').write_text(json.dumps({'model_type': 'qwen_drive',
        'vlm_config': {'model_type': 'qwen3_5', 'architectures': ['Qwen3_5ForConditionalGeneration']}}))
    (source / 'tokenizer_config.json').write_text('{}')
    header = {
        'vlm.model.visual.patch_embed.proj.weight': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]},
        'planning_expert.weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [4, 8]},
        'vlm.model.language_model.norm.weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [8, 12]},
    }
    encoded = json.dumps(header).encode()
    encoded += b' ' * (-len(encoded) % 8)
    content = struct.pack('<Q', len(encoded)) + encoded + b'abcdSKIPefgh'
    (source / 'model.safetensors').write_bytes(content)
    output = tmp_path / 'adapter'
    summary = script.export(source, output)
    assert summary['tensors'] == 2 and summary['tensor_bytes'] == 8
    result, offset = script.read_header(output / 'model.safetensors')
    assert result['model.visual.patch_embed.proj.weight']['data_offsets'] == [0, 4]
    assert result['model.language_model.norm.weight']['data_offsets'] == [4, 8]
    assert (output / 'model.safetensors').read_bytes()[offset:] == b'abcdefgh'
    assert (source / 'model.safetensors').read_bytes() == content
    assert json.loads((output / 'config.json').read_text())['model_type'] == 'qwen3_5'
    assert (output / 'tokenizer_config.json').read_text() == '{}'
    with pytest.raises(FileExistsError):
        script.export(source, output)


def test_vllm_uses_official_prompt_and_checks_expansion(monkeypatch):
    vllm = ModuleType('vllm')
    vllm.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, 'vllm', vllm)
    package = ModuleType('qwen_drive')
    package.__path__ = []
    module = ModuleType('qwen_drive.modeling_qwen_drive')
    module.VQA_DECODE_DEFAULTS = {'temperature': 0.01, 'top_k': 1, 'top_p': 0.001,
                                'repetition_penalty': 1.0, 'presence_penalty': 0.0, 'seed': 3407}
    monkeypatch.setitem(sys.modules, 'qwen_drive', package)
    monkeypatch.setitem(sys.modules, 'qwen_drive.modeling_qwen_drive', module)
    official = [2, 55, 55, 3, 2, 55, 3, 9]
    calls = []
    outputs = SimpleNamespace(prompt_token_ids=official,
                              outputs=[SimpleNamespace(text='<think>x</think>{"ok":1}',
                                                       token_ids=[4, 5], finish_reason='stop')])
    def generate(requests, params, **kwargs):
        calls.append((requests, params))
        return [outputs]
    backend = VLLMQwenDrive.__new__(VLLMQwenDrive)
    backend.processor = SimpleNamespace(image_token_id=55,
        encode_vqa=lambda *args, **kwargs: {'input_ids': torch.tensor([official])})
    backend.llm = SimpleNamespace(generate=generate)
    assert backend.generate_text(['left', 'right'], 'road?', max_new_tokens=384).text == '{"ok":1}'
    assert calls[0][0][0]['prompt_token_ids'] == [2, 55, 3, 2, 55, 3, 9]
    assert calls[0][1]['max_tokens'] == 384 and calls[0][1]['top_k'] == 1
    assert backend.last_metrics['prompt_ids_exact'] is True
    outputs.prompt_token_ids = [1, 2, 3]
    with pytest.raises(QwenDriveInputMismatchError):
        backend.generate_text(['left', 'right'], 'road?', max_new_tokens=384)


def test_collapse_keeps_separate_image_spans():
    assert collapse_image_tokens([55, 55, 3, 55, 55, 3], 55) == [55, 3, 55, 3]


def test_probe_compares_usable_semantics_not_only_json_syntax(tmp_path):
    probe = load_script('probe_p7_qwendrive_backend')
    proposal = {'scene_rules': {'signal': 'green', 'stop_sign': 'no',
                 'pedestrian_conflict': 'no', 'vehicle_conflict': 'no'},
                'intents': [{'maneuver': 'keep_lane', 'obligations': [], 'confidence': 0.8}]}
    row = {'image': 'a.jpg', 'navigation': 'follow_lane', 'speed': 3,
           'seconds': 10, 'raw_text': json.dumps(proposal), 'error': None}
    hf, engine = tmp_path / 'hf.jsonl', tmp_path / 'vllm.jsonl'
    hf.write_text(json.dumps(row) + '\n')
    engine.write_text(json.dumps({**row, 'seconds': 2}) + '\n')
    result = probe.compare(hf, engine)
    assert result['semantic_equal_and_valid'] == 1
    engine.write_text(json.dumps({**row, 'raw_text': '{}'}) + '\n')
    assert probe.compare(hf, engine)['semantic_equal_and_valid'] == 0


def test_live_service_does_not_turn_input_mismatch_into_fallback():
    import socket
    from threading import Thread
    import numpy as np
    from lead.inference.qwendrive_intent_client import QwenDriveIntentClient
    from lead.inference.qwendrive_intent_service import serve
    client_socket, server_socket = socket.socketpair()
    client_socket.settimeout(5)
    client = QwenDriveIntentClient.__new__(QwenDriveIntentClient)
    client.sock, client.requests = client_socket, 0
    observed = []
    class Model:
        def generate_text(self, *args, **kwargs):
            raise QwenDriveInputMismatchError('bad prompt expansion')
    def worker():
        try:
            serve(Model(), server_socket, None, analysis_tokens=512, answer_tokens=384)
        except QwenDriveInputMismatchError as exc:
            observed.append(str(exc))
        finally:
            server_socket.close()
    thread = Thread(target=worker)
    thread.start()
    try:
        with pytest.raises(ConnectionError):
            client.generate_rgb(np.zeros((384,1152,3), dtype=np.uint8), 'follow_lane', 0.0)
    finally:
        client.close()
        thread.join(timeout=5)
    assert observed == ['bad prompt expansion']
    assert not thread.is_alive()
