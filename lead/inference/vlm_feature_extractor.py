"""Shared Qwen-VL feature extraction for visual intent (P4/P5).

Single source of truth for turning a front-camera image into the cached
``(h', w', D)`` image-token hidden states that ``VLMIntentDecoder`` consumes.

Both the offline extractor (``scripts/p4/extract_vlm_features.py``) and the
closed-loop agent (``lead/inference/sensor_agent.py``) call ``extract_vlm_hidden``
so the exact prompt / crop / token-grab logic never drifts between train and eval.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

# Command-conditioned prompt (P4): the nav command is placed BEFORE the image so the
# causal image-token hidden states get conditioned on it.
PROMPT_COMMAND = (
    "You are the motion planner of a car. Navigation command: {cmd}. "
    "Look at the front camera image and determine where the ego vehicle should drive "
    "to follow this command."
)

# Command-agnostic prompt (P5): no command -> the image-token hidden states encode ALL
# drivable directions (the feasible set), leaving the command to be resolved downstream.
PROMPT_DRIVABLE = (
    "You are the motion planner of a car. Look at the front camera image and identify "
    "every direction the car could drive from here -- all lanes and turns that are "
    "drivable ahead, without committing to a single one."
)


def load_qwen_vl(model_path: str, device: str = "cuda:0"):
    """Load a frozen Qwen-VL model + processor for feature extraction.

    Returns:
        (model, processor, image_token_id, spatial_merge_size)
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()
    image_token_id = model.config.image_token_id
    merge = getattr(model.config.vision_config, "spatial_merge_size", 2)
    return model, processor, image_token_id, merge


def crop_front(image: Image.Image, front_frac: tuple[float, float]) -> Image.Image:
    """Crop the front camera out of a multi-camera horizontal strip by width fraction."""
    w, h = image.size
    f0, f1 = front_frac
    return image.crop((int(w * f0), 0, int(w * f1), h))


@torch.no_grad()
def extract_vlm_hidden(
    front_image: Image.Image,
    model,
    processor,
    image_token_id: int,
    merge: int,
    prompt_mode: str = "drivable",
    command: str = "LANEFOLLOW",
    layer: int = -1,
    device: str = "cuda:0",
) -> npt.NDArray:
    """Run one frozen Qwen-VL prefill on a front-camera image and return image-token hidden states.

    Args:
        front_image: PIL RGB image of the FRONT camera (already cropped out of the strip).
        model, processor, image_token_id, merge: from ``load_qwen_vl``.
        prompt_mode: ``"drivable"`` (P5, command-agnostic) or ``"command"`` (P4).
        command: nav command word, used only when ``prompt_mode == "command"``.
        layer: which ``hidden_states`` layer to grab (-1 = last).

    Returns:
        ``(h', w', D)`` float16 array of image-token hidden states (typically (12, 12, 2560)).

    Raises:
        ValueError: if the recovered token grid does not match the number of image tokens.
    """
    prompt = (
        PROMPT_DRIVABLE if prompt_mode == "drivable"
        else PROMPT_COMMAND.format(cmd=command)
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": front_image},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(device)

    out = model(**inputs, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[layer][0]  # (seq, D)
    mask = inputs["input_ids"][0] == image_token_id
    img_hs = hs[mask]  # (n_img, D)

    grid = inputs["image_grid_thw"][0].tolist()  # [t, h, w]
    h2, w2 = grid[1] // merge, grid[2] // merge
    if h2 * w2 != img_hs.shape[0]:
        raise ValueError(
            f"token/grid mismatch: {img_hs.shape[0]} image tokens vs {h2}x{w2} grid"
        )
    return img_hs.reshape(h2, w2, -1).to(torch.float16).cpu().numpy()
