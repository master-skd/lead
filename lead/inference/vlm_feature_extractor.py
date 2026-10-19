"""Shared Qwen-VL feature extraction for visual intent (P4/P5).

Single source of truth for turning a front-camera image into the cached
``(h', w', D)`` image-token hidden states that ``VLMIntentDecoder`` consumes.

Both the offline extractor (``scripts/p4/extract_vlm_features.py``) and the
closed-loop agent (``lead/inference/sensor_agent.py``) call ``extract_vlm_hidden``
so the exact prompt / crop / token-grab logic never drifts between train and eval.
"""

from __future__ import annotations

from contextlib import nullcontext

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

# P6 3-camera prompt. MUST match scripts/p4/extract_vlm_p6_3cam.py verbatim so live
# closed-loop features are identical to the training cache (train/eval consistency).
PROMPT_3CAM_DRIVABLE = (
    "You are the motion planner of a car. The image is a panorama stitched from three "
    "front-facing cameras: front-left, front, and front-right. Look across the whole "
    "panorama and identify every direction the car could drive from here -- all lanes and "
    "turns that are drivable ahead, including to the sides, without committing to one."
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


def prepare_p6_batch_prompt(processor) -> str:
    """Render the fixed P6 chat prompt once; image pixels are supplied separately."""
    return processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "text", "text": PROMPT_3CAM_DRIVABLE},
            {"type": "image"},
        ]}],
        tokenize=False, add_generation_prompt=True,
    )


def prepare_p6_batch_inputs(
    images: list[Image.Image], processor, *, prompt: str | None = None,
    image_device: str | None = None,
):
    """Prepare a P6 batch, preserving the original template path as the reference."""
    if prompt is not None:
        kwargs = {"padding": len(images) > 1, "return_tensors": "pt"}
        if image_device is not None:
            kwargs["images_kwargs"] = {"device": image_device}
        return processor(text=[prompt] * len(images), images=images, **kwargs)

    conversations = [
        [{"role": "user", "content": [
            {"type": "text", "text": PROMPT_3CAM_DRIVABLE},
            {"type": "image", "image": image},
        ]}]
        for image in images
    ]
    return processor.apply_chat_template(
        conversations[0] if len(images) == 1 else conversations,
        tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
        processor_kwargs={"padding": len(images) > 1},
    )


def check_p6_batch_input_parity(reference, candidate) -> None:
    """Fail closed if a faster processor path changes Qwen's model inputs."""
    if set(reference) != set(candidate):
        raise ValueError(f"Qwen input keys differ: {set(reference)} != {set(candidate)}")
    for key in reference:
        left, right = reference[key], candidate[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"Qwen {key} shape/dtype differs")
        right = right.to(left.device)
        if key == "pixel_values":
            if not torch.allclose(left, right, atol=1e-6, rtol=0):
                error = (left - right).abs().max().item()
                raise ValueError(f"Qwen pixel_values differ: max_abs_error={error}")
        elif not torch.equal(left, right):
            raise ValueError(f"Qwen {key} differs")


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
        PROMPT_3CAM_DRIVABLE if prompt_mode == "3cam_drivable"
        else PROMPT_DRIVABLE if prompt_mode == "drivable"
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


@torch.no_grad()
def extract_vlm_hidden_batch(
    images: list[Image.Image], model, processor, image_token_id: int, merge: int,
    *, prompt_mode: str = "3cam_drivable", device: str = "cuda:0", timing=None,
    prepared_prompt: str | None = None, image_preprocess_device: str | None = None,
) -> npt.NDArray:
    """One Qwen prefill for an equal-sized batch, matching the P6 fp16 cache boundary."""
    if not images:
        raise ValueError("empty VLM image batch")
    if prompt_mode != "3cam_drivable":
        raise ValueError("batched extraction currently supports 3cam_drivable only")
    def phase(name, gpu=False):
        return timing.phase(name, gpu=gpu) if timing is not None else nullcontext()

    with phase("preprocess", gpu=image_preprocess_device is not None):
        inputs = prepare_p6_batch_inputs(
            images, processor, prompt=prepared_prompt,
            image_device=image_preprocess_device,
        )
    with phase("inputs_to_gpu", gpu=True):
        inputs = inputs.to(device)
    with phase("qwen_forward", gpu=True):
        output = model(**inputs, output_hidden_states=True, use_cache=False)
    with phase("extract_and_to_cpu", gpu=True):
        hidden = output.hidden_states[-1]
        if len(inputs["image_grid_thw"]) != len(images):
            raise ValueError("Qwen image grid count does not match request")
        features = []
        for i, grid in enumerate(inputs["image_grid_thw"].tolist()):
            h, w = grid[1] // merge, grid[2] // merge
            tokens = hidden[i][inputs["input_ids"][i] == image_token_id]
            if (h, w, tokens.shape[0]) != (12, 36, 432):
                raise ValueError(f"unexpected image-token grid: {(h, w, tokens.shape[0])}")
            features.append(tokens.reshape(h, w, -1).to(torch.float16).cpu().numpy())
        return np.stack(features)
