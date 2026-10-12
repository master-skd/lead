import json
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from lead.common.stage_timing import StageTimingRecorder
from lead.inference.vlm_feature_extractor import extract_vlm_hidden_batch
from scripts.p4.summarize_p6_joint_timing import select_records


def test_disabled_timing_does_not_sync_or_write(tmp_path):
    recorder = StageTimingRecorder(tmp_path / "absent", "train", 0)
    recorder.sync = lambda: (_ for _ in ()).throw(AssertionError("unexpected sync"))
    timing = recorder.start("train_step", index=0)
    with timing.phase("forward", gpu=True):
        pass
    timing.finish()
    assert not (tmp_path / "absent").exists()


def test_sampling_and_gpu_boundaries(tmp_path):
    recorder = StageTimingRecorder(tmp_path, "train", 2, enabled=True, every=20)
    assert not recorder.start("step", index=6).active
    assert recorder.start("step", index=6, force=True).active
    syncs = []
    recorder.sync = lambda: syncs.append(True)
    timing = recorder.start("train_step", index=20, epoch=1, step=20)
    with timing.phase("data_wait"):
        pass
    with timing.phase("forward", gpu=True):
        pass
    timing.finish()
    row = json.loads(recorder.path.read_text())
    assert len(syncs) == 2
    assert row["rank"] == 2 and row["epoch"] == 1
    assert row["total_ms"] >= sum(row["stages_ms"].values())
    assert row["stage_start_ms"]["forward"] >= row["stage_start_ms"]["data_wait"]


def test_summary_aligns_requests_and_excludes_warmup_validation_and_old_launches():
    base = dict(run_id="new", timestamp=2, rank=0)
    rows = [
        dict(base, event="train_step", step=0, request_index=0),
        dict(base, event="train_step", step=20, request_index=20),
        dict(base, event="qwen_batch", request_index=0),
        dict(base, event="qwen_batch", request_index=20),
        dict(base, event="qwen_batch", request_index=40),  # validation
        dict(
            base,
            event="train_step",
            step=20,
            request_index=20,
            run_id="old",
            timestamp=1,
        ),
    ]
    selected, run_id = select_records(rows)
    assert run_id == "new"
    assert [(row["event"], row["request_index"]) for row in selected] == [
        ("train_step", 20),
        ("qwen_batch", 20),
    ]


def test_feature_timing_preserves_batch_output(tmp_path):
    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return Inputs(
                input_ids=torch.ones(2, 432, dtype=torch.long),
                image_grid_thw=torch.tensor([[1, 24, 72], [1, 24, 72]]),
            )

    hidden = torch.arange(2 * 432 * 4).reshape(2, 432, 4).float()

    def model(**kwargs):
        return SimpleNamespace(hidden_states=(hidden,))

    images = [Image.new("RGB", (1152, 384)) for _ in range(2)]
    recorder = StageTimingRecorder(tmp_path, "qwen", 0, enabled=True, device=None)
    timing = recorder.start("qwen_batch", index=0)
    actual = extract_vlm_hidden_batch(
        images, model, Processor(), 1, 2, device="cpu", timing=timing
    )
    timing.finish()
    reference = extract_vlm_hidden_batch(images, model, Processor(), 1, 2, device="cpu")
    np.testing.assert_array_equal(actual, reference)
    assert actual.shape == (2, 12, 36, 4)
    assert set(json.loads(recorder.path.read_text())["stages_ms"]) == {
        "preprocess",
        "inputs_to_gpu",
        "qwen_forward",
        "extract_and_to_cpu",
    }
