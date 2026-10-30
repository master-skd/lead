"""Offline JSON generation must preserve online semantic inputs and fail on gaps."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from lead.tfv6.semantic_intent_cache import IntentCache, build_index, image_key, recover_jsonl
from lead.tfv6.semantic_intent_slots import service_results_to_tensors


def proposal(maneuver='keep_lane'):
    return {'scene_rules': {'signal': 'red', 'stop_sign': 'no',
                           'pedestrian_conflict': 'no', 'vehicle_conflict': 'uncertain'},
            'intents': [{'maneuver': maneuver, 'obligations': ['stop_for_red'],
                         'confidence': 0.8, 'evidence': 'red light'}]}


def fixture_cache(tmp_path, rows=None, manifest_rows=None):
    rows = rows or [{'image_key': 'A/route/rgb/0001.jpg', 'navigation': 'follow_lane',
                     'speed': 3.0, 'raw_text': json.dumps(proposal()), 'error': None}]
    manifest_rows = manifest_rows or [{k: row[k] for k in ('image_key', 'navigation', 'speed')}
                                     for row in rows]
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in manifest_rows))
    shards = tmp_path / 'shards'
    shards.mkdir()
    signature = {'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (shards / 'shard-00000.jsonl').write_text(
        json.dumps({'type': 'metadata', 'signature': signature}) + '\n' +
        ''.join(json.dumps(row) + '\n' for row in rows))
    return manifest, shards, tmp_path / 'intents.sqlite'


def test_cached_and_live_semantics_match_after_moving_data_root(tmp_path):
    manifest, shards, db = fixture_cache(tmp_path)
    summary = build_index(manifest, shards, db)
    assert summary['valid_branch_histogram_before_fallback'] == {1: 1}
    cache = IntentCache(str(db), str(manifest))
    try:
        cached = cache.generate_batch(['/another/machine/A/route/rgb/0001.jpg'], ['follow_lane'], [3.0])
        live = [{'raw_text': json.dumps(proposal()), 'error': None}]
        for key, expected in service_results_to_tensors(live, ['follow_lane'], capacity=6).items():
            assert torch.equal(expected, service_results_to_tensors(cached, ['follow_lane'], capacity=6)[key])
        with pytest.raises(KeyError, match='missing frame'):
            cache.generate_batch(['A/route/rgb/0002.jpg'], ['follow_lane'], [3.0])
        with pytest.raises(ValueError, match='differs from training'):
            cache.generate_batch(['A/route/rgb/0001.jpg'], ['turn_left'], [3.0])
        with pytest.raises(ValueError, match='differs from training'):
            cache.generate_batch(['A/route/rgb/0001.jpg'], ['follow_lane'], [4.0])
    finally:
        cache.close()


def test_index_refuses_incomplete_manifest(tmp_path):
    manifest, shards, db = fixture_cache(tmp_path, manifest_rows=[
        {'image_key': 'A/route/rgb/missing.jpg', 'navigation': 'follow_lane', 'speed': 3.0}])
    with pytest.raises(ValueError, match='unfinished generation'):
        build_index(manifest, shards, db)
    assert not db.exists()


def test_jsonl_repairs_only_interrupted_last_write(tmp_path):
    path = tmp_path / 'shard.jsonl'
    path.write_bytes(b'{"ok": 1}\n{"partial"')
    assert list(recover_jsonl(path)) == [{'ok': 1}]
    assert path.read_bytes() == b'{"ok": 1}\n'
    path.write_bytes(b'broken\n')
    with pytest.raises(json.JSONDecodeError):
        list(recover_jsonl(path))


def test_failed_generation_counts_zero_real_branches(tmp_path):
    row = {'image_key': 'A/route/rgb/0001.jpg', 'navigation': 'follow_lane',
           'speed': 3.0, 'raw_text': '', 'error': 'generation failed'}
    manifest, shards, db = fixture_cache(tmp_path, rows=[row])
    summary = build_index(manifest, shards, db)
    assert summary['fallback'] == 1
    assert summary['valid_branch_histogram_before_fallback'] == {0: 1}


def test_generation_resume_and_prompt_mismatch(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('p7_generate', 'scripts/p7/generate_p7_intents.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    root = tmp_path / 'data'
    image = root / 'A/route/rgb/0001.jpg'
    image.parent.mkdir(parents=True)
    Image.new('RGB', (1152, 384)).save(image)
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps({'image_key': image_key(str(image)),
                                    'navigation': 'follow_lane', 'speed': 3.0}) + '\n')
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    (model_dir / 'config.json').write_text('{}')
    calls = []
    class Model:
        def generate_text(self, images, prompt, **kwargs):
            calls.append(prompt)
            return SimpleNamespace(text='road analysis' if len(calls) % 2 else json.dumps(proposal()))
    monkeypatch.setattr(script, 'load_vqa_model', lambda *args: Model())
    args = SimpleNamespace(model=str(model_dir), manifest=manifest, output_dir=tmp_path / 'output',
                           data_root=root, device='cpu', shard=0, num_shards=1,
                           analysis_tokens=512, answer_tokens=384, retry_failed=False)
    script.generate(args)
    assert len(calls) == 2
    # Interrupted tail is repaired, then completed frames are skipped without loading Qwen.
    with (args.output_dir / 'shard-00000.jsonl').open('a') as stream:
        stream.write('{"partial"')
    monkeypatch.setattr(script, 'load_vqa_model', lambda *args: pytest.fail('resume loaded model'))
    script.generate(args)
    assert len(calls) == 2
    args.answer_tokens = 200
    with pytest.raises(ValueError, match='settings changed'):
        script.generate(args)


def test_cached_launcher_needs_no_qwen_environment(tmp_path):
    import os
    import shutil
    import subprocess
    project = tmp_path / 'project'
    scripts = project / 'scripts/p7'
    scripts.mkdir(parents=True)
    for name in ('train_p7_cached.sh', 'train_p7_hf_online.sh'):
        shutil.copyfile(Path('scripts/p7') / name, scripts / name)
    (project / 'data/carla_leaderboard2/data').mkdir(parents=True)
    (project / 'data/carla_leaderboard2/buckets').mkdir()
    checkpoint = project / 'posttrain'
    checkpoint.mkdir()
    (checkpoint / 'model_0030.pth').touch()
    (checkpoint / 'config.json').write_text('{}')
    manifest = project / 'manifest.jsonl'
    manifest.write_text('{}\n')
    cache = project / 'intents.sqlite'
    cache.touch()  # Launcher checks presence; the Python reader validates content.
    binary = project / 'bin'
    binary.mkdir()
    for name, body in [('python', '#!/bin/sh\nexit 0\n'),
                       ('torchrun', '#!/bin/sh\nprintf "%s\\n" "$LEAD_TRAINING_CONFIG"\n')]:
        path = binary / name
        path.write_text(body)
        path.chmod(0o755)
    result = subprocess.run(['bash', str(scripts / 'train_p7_cached.sh')],
        env={**os.environ, 'P7_LEAD_PYTHON': str(binary / 'python'),
             'P7_POSTTRAIN': str(checkpoint / 'model_0030.pth'),
             'P7_INTENT_CACHE': str(cache), 'P7_MANIFEST': str(manifest),
             'P7_QWEN_PYTHON': '/missing/python', 'P7_QWEN_MODEL': '/missing/model',
             'P7_LEAD_LIBNVRTC_DIR': '/missing/nvrtc', 'CUDA_VISIBLE_DEVICES': '0,1',
             'P7_LOGDIR': str(project / 'logs')},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert f'p7_intent_cache={cache}' in result.stdout
    assert f'p7_intent_manifest={manifest}' in result.stdout
    assert 'multimodal_target_speed=true' in result.stdout
    assert not list((project / 'logs').glob('qwendrive_rank*.log'))
