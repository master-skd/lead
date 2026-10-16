"""Preflight regressions for the single-checkpoint P6 Bench2Drive launcher."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/p4/eval_p6_joint_b2d.sh"


def _fixture(tmp_path, *, multimodal_speed=False):
    training = tmp_path / "training"
    training.mkdir()
    checkpoint = training / "model_0019.pth"
    checkpoint.write_bytes(b"preflight does not load weights")
    intent = tmp_path / "intent.pth"
    intent.write_bytes(b"intent")
    (training / "config.json").write_text(json.dumps({
        "use_vlm_intent": True,
        "vlm_3cam": True,
        "multimodal_planner": True,
        "joint_scorer_replaces_conf": True,
        "multimodal_target_speed": multimodal_speed,
        "vlm_intent_ckpt": str(intent),
    }))
    carla = tmp_path / "carla"
    carla.mkdir()
    launcher = carla / "CarlaUE4.sh"
    launcher.write_text("#!/bin/sh\n")
    launcher.chmod(0o755)
    model = tmp_path / "qwen"
    model.mkdir()
    env = os.environ.copy()
    env.update({
        "CARLA_ROOT": str(carla),
        "P6_B2D_LEAD_PYTHON": sys.executable,
        "P6_B2D_QWEN_PYTHON": sys.executable,
        "P6_B2D_QWEN_MODEL": str(model),
        "P6_B2D_EVAL_CHECKPOINT_ROOT": str(tmp_path / "eval"),
        "P6_B2D_PREFLIGHT_ONLY": "1",
    })
    return checkpoint, env


def test_p6_b2d_preflight_exposes_exactly_one_model(tmp_path):
    checkpoint, env = _fixture(tmp_path)
    result = subprocess.run(
        ["bash", str(SCRIPT), str(checkpoint), "0 1", "pilot"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    eval_dir = tmp_path / "eval/pilot"
    assert sorted(p.name for p in eval_dir.iterdir()) == ["config.json", "model.pth"]
    assert (eval_dir / "model.pth").resolve() == checkpoint


def test_p6_b2d_preflight_rejects_k_speed_checkpoint(tmp_path):
    checkpoint, env = _fixture(tmp_path, multimodal_speed=True)
    result = subprocess.run(
        ["bash", str(SCRIPT), str(checkpoint), "0", "wrong_model"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "K-route/one-speed" in result.stderr
    assert not (tmp_path / "eval/wrong_model").exists()
