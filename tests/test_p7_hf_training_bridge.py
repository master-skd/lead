"""CPU checks for the Qwen-Drive text-to-P7 training handoff."""

import json
import socket
from threading import Thread
from types import SimpleNamespace

from PIL import Image

from lead.inference.qwendrive_intent_client import _receive_json, _send_json
from lead.inference.qwendrive_intent_service import _views, serve
from lead.tfv6.semantic_intent_slots import service_results_to_tensors


def test_service_and_parser_preserve_branch_obligations(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    image = root / "frame.jpg"
    Image.new("RGB", (1152, 384), (20, 30, 40)).save(image)
    proposal = {
        "scene_rules": {"signal": "red", "stop_sign": "no",
                        "pedestrian_conflict": "no", "vehicle_conflict": "uncertain"},
        "intents": [{"maneuver": "keep_lane", "obligations": ["stop_for_red"],
                     "confidence": 0.8, "evidence": "signal ahead"}],
    }

    class FakeModel:
        def __init__(self):
            self.calls = []

        def generate_text(self, images, prompt, **kwargs):
            self.calls.append((len(images), prompt, kwargs))
            return SimpleNamespace(text="clear road" if len(self.calls) == 1
                                   else json.dumps(proposal))

    model = FakeModel()
    client, server = socket.socketpair()
    thread = Thread(target=serve, args=(model, server, root.resolve()),
                    kwargs={"analysis_tokens": 512, "answer_tokens": 384})
    thread.start()
    try:
        _send_json(client, {"samples": [{"image": str(image), "command": "follow_lane",
                                          "speed": 4.0}]})
        result = _receive_json(client)["samples"]
    finally:
        client.close()
        thread.join(timeout=5)
        server.close()
    assert not thread.is_alive()
    assert len(model.calls) == 2
    assert all(call[0] == 3 for call in model.calls)
    assert "clear road" in model.calls[1][1]
    tensors = service_results_to_tensors(result, ["follow_lane"], capacity=6)
    assert tensors["semantic_intent_valid"].sum().item() == 1
    assert not tensors["semantic_intent_fallback"].item()
    assert tensors["semantic_intent_obligations"][0, 0].sum().item() == 1


def test_empty_or_broken_response_uses_nonsemantic_slot_zero():
    result = [{"raw_text": "not json", "error": None},
              {"raw_text": "", "error": "generation failed"}]
    tensors = service_results_to_tensors(result, ["straight", "turn_left"], capacity=6)
    assert tensors["semantic_intent_fallback"].tolist() == [True, True]
    assert tensors["semantic_intent_valid"].tolist() == [
        [True, False, False, False, False, False],
        [True, False, False, False, False, False],
    ]


def test_views_rejects_outside_media_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (1152, 384)).save(outside)
    try:
        _views(str(outside), root)
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("media root must be enforced")
