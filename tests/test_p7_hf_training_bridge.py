"""CPU checks for the Qwen-Drive text-to-P7 training handoff."""

import json
import socket
from threading import Event, Thread
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from lead.inference.qwendrive_intent_client import (
    QwenDriveIntentPrefetcher, _receive_json, _send_json,
)
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


def test_p7_prefetch_overlaps_ordered_batches_and_preserves_metadata():
    entered = Event()
    release = Event()

    class Client:
        def __init__(self):
            self.calls = []

        def generate_batch(self, paths, commands, speeds):
            self.calls.append((paths, commands, speeds))
            if len(self.calls) == 2:
                entered.set()
                assert release.wait(timeout=5)
            return [{"raw_text": path, "error": None} for path in paths]

    def batch(path, command_id, speed):
        return {
            "p7_image_path": [path],
            "command": torch.nn.functional.one_hot(torch.tensor([command_id]), 6),
            "speed": torch.tensor([speed]),
        }

    client = Client()
    batches = iter([batch("a.jpg", 3, 4.0), batch("b.jpg", 0, 5.0)])
    with QwenDriveIntentPrefetcher(client) as prefetch:
        assert prefetch.advance(batches)
        assert prefetch.current_navigation == ["follow_lane"]
        assert prefetch.result()[0]["raw_text"] == "a.jpg"
        assert prefetch.advance(batches)
        assert prefetch.current_navigation == ["turn_left"]
        assert entered.wait(timeout=5)  # Next VQA runs while the current batch trains.
        release.set()
        assert prefetch.result()[0]["raw_text"] == "b.jpg"
        assert not prefetch.advance(batches)
    assert client.calls == [
        (["a.jpg"], ["follow_lane"], [4.0]),
        (["b.jpg"], ["turn_left"], [5.0]),
    ]


def test_p7_prefetch_rejects_bad_command_before_request():
    class Client:
        def generate_batch(self, *_):
            raise AssertionError("invalid command must never reach the service")

    with QwenDriveIntentPrefetcher(Client()) as prefetch:
        with pytest.raises(ValueError, match="six-way one-hot"):
            prefetch.advance(iter([{
                "p7_image_path": ["a.jpg"],
                "command": torch.zeros(1, 6),
                "speed": torch.tensor([0.0]),
            }]))
