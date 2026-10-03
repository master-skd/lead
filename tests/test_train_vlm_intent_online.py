import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lead.inference.vlm_feature_extractor import PROMPT_3CAM_DRIVABLE as ONLINE_PROMPT
from scripts.p4.extract_vlm_p6_3cam import PROMPT_3CAM_DRIVABLE as CACHE_PROMPT
from scripts.p4.train_vlm_intent_online import OnlineIntentDataset, qwen_image_tokens


def test_online_and_cached_prompts_match():
    assert ONLINE_PROMPT == CACHE_PROMPT


def test_online_dataset_reads_only_rgb_and_lanegraph_label(tmp_path: Path):
    image = tmp_path / "rgb.jpg"
    Image.new("RGB", (1152, 384), (10, 20, 30)).save(image)
    labels = tmp_path / "labels" / "scenario" / "route"
    labels.mkdir(parents=True)
    np.save(labels / "0001.npy", np.ones((1, 320, 384), dtype=np.float16))
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({
        "src": str(image), "scenario": "scenario", "route": "route", "frame": "0001",
    }) + "\n")

    sample_image, sample_label = OnlineIntentDataset(manifest, tmp_path / "labels")[0]
    assert sample_image.size == (1152, 384)
    assert sample_label.shape == (1, 320, 384)
    assert sample_label.dtype == torch.float32


def test_online_dataset_remaps_rgb_root(tmp_path: Path):
    rgb_root = tmp_path / "elsewhere" / "data" / "carla_leaderboard2" / "data"
    image = rgb_root / "scenario" / "route" / "rgb" / "0001.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (1152, 384)).save(image)
    labels = tmp_path / "labels" / "scenario" / "route"
    labels.mkdir(parents=True)
    np.save(labels / "0001.npy", np.zeros((1, 320, 384), dtype=np.float16))
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({
        "src": "data/carla_leaderboard2/data/scenario/route/rgb/0001.jpg",
        "scenario": "scenario", "route": "route", "frame": "0001",
    }) + "\n")

    dataset = OnlineIntentDataset(manifest, tmp_path / "labels", rgb_root=rgb_root)
    assert dataset.samples[0][0] == image
    assert dataset[0][0].size == (1152, 384)


def test_online_qwen_tokens_keep_only_image_positions_and_fp16_boundary():
    class Inputs(dict):
        def to(self, _device):
            return self

    class Processor:
        def apply_chat_template(self, conversations, **kwargs):
            count = 1 if len(conversations) == 1 and isinstance(conversations[0], dict) else len(conversations)
            ids = torch.full((count, 435), 7)
            ids[:, 1:433] = 99
            return Inputs(
                input_ids=ids,
                image_grid_thw=torch.tensor([[1, 24, 72]] * count),
            )

    class Model:
        def __call__(self, input_ids, **_kwargs):
            hidden = torch.arange(input_ids.shape[0] * 435 * 2560, dtype=torch.float32)
            hidden = hidden.reshape(input_ids.shape[0], 435, 2560) / 10000
            return type("Result", (), {"hidden_states": (hidden,)})()

    tokens = qwen_image_tokens(
        [Image.new("RGB", (1152, 384)) for _ in range(3)],
        Model(), Processor(), image_token_id=99, merge=2,
        microbatch=2, device="cpu",
    )
    assert tokens.shape == (3, 12, 36, 2560)
    assert tokens.dtype == torch.float32
    assert not tokens.is_inference()
    assert tokens[0, 0, 0, 0] == torch.tensor(2560 / 10000).half().float()
