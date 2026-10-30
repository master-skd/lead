"""Frozen Qwen-Drive HF VQA service; no Planning Expert weights are loaded."""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path
from functools import lru_cache

from PIL import Image

from lead.inference.qwendrive_intent_client import (
    NAV_COMMANDS, _receive_json, _send_json,
)


def _views(path: str, media_root: Path) -> list[Image.Image]:
    image = Path(path).resolve(strict=True)
    if not image.is_relative_to(media_root):
        raise ValueError("image path lies outside the allowed CARLA data root")
    with Image.open(image) as source:
        strip = source.convert("RGB")
    if strip.size != (1152, 384):
        raise ValueError(f"expected 384x1152 three-camera strip, got {strip.size}")
    width = strip.width // 3
    # Match the reviewed probe: front, front-left, front-right.
    return [strip.crop((width, 0, 2 * width, strip.height)),
            strip.crop((0, 0, width, strip.height)),
            strip.crop((2 * width, 0, 3 * width, strip.height))]


@lru_cache(maxsize=1)
def intent_prompts():
    # The reviewed P7 prompt remains the single source of truth.
    probe_dir = Path(__file__).resolve().parents[2] / "scripts" / "p7"
    sys.path.insert(0, str(probe_dir))
    from probe_qwendrive_rule_intents_cpu import analysis_prompt, final_prompt
    return analysis_prompt, final_prompt


def generate_intent(model, image: str, command: str, speed: float, media_root: Path,
                    *, analysis_tokens: int, answer_tokens: int) -> dict:
    analysis_prompt, final_prompt = intent_prompts()
    if command not in NAV_COMMANDS:
        raise ValueError("invalid navigation command")
    images = _views(image, media_root)
    analysis = model.generate_text(
        images, analysis_prompt(command, speed), max_new_tokens=analysis_tokens,
    ).text
    answer = model.generate_text(
        images, final_prompt(command, speed, analysis), max_new_tokens=answer_tokens,
    ).text
    return {"raw_text": answer, "analysis": analysis, "error": None}


def load_vqa_model(model_path: str, device: str):
    import torch
    from qwen_drive import QwenDriveForPlanning

    model = QwenDriveForPlanning.from_pretrained(
        model_path, dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).eval()
    # Only the VLM is used; the uninitialized Planning Expert stays on CPU.
    model.vlm.to(device).eval()
    return model


def serve(model, connection: socket.socket, media_root: Path, *, analysis_tokens: int,
          answer_tokens: int) -> None:

    while True:
        try:
            request = _receive_json(connection)
        except ConnectionError:
            return
        samples = request.get("samples") if isinstance(request, dict) else None
        if not isinstance(samples, list) or not 0 < len(samples) <= 256:
            raise ValueError("invalid Qwen-Drive request batch")
        results = []
        for item in samples:
            try:
                results.append(generate_intent(
                    model, item["image"], item["command"], float(item["speed"]),
                    media_root, analysis_tokens=analysis_tokens, answer_tokens=answer_tokens,
                ))
            except Exception as exc:
                sample_path = item.get("image", "<unknown>") if isinstance(item, dict) else "<invalid>"
                print(f"[qwendrive] sample={sample_path} "
                      f"failed: {type(exc).__name__}: {exc}", flush=True)
                results.append({"raw_text": "", "error": f"{type(exc).__name__}: {exc}"})
        _send_json(connection, {"samples": results})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--analysis-tokens", type=int, default=512)
    parser.add_argument("--answer-tokens", type=int, default=384)
    args = parser.parse_args()
    if args.analysis_tokens < 1 or args.answer_tokens < 1:
        raise ValueError("generation token limits must be positive")
    media_root = Path(args.media_root).resolve(strict=True)

    model = load_vqa_model(args.model, args.device)
    path = Path(args.socket)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing socket: {path}")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
        server.listen(1)
        print(f"[qwendrive] ready on {path}; VQA only", flush=True)
        try:
            while True:
                connection, _ = server.accept()
                with connection:
                    try:
                        serve(model, connection, media_root,
                              analysis_tokens=args.analysis_tokens,
                              answer_tokens=args.answer_tokens)
                    except (ConnectionError, BrokenPipeError):
                        pass
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
