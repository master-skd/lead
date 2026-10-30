"""CPU contracts for real batch collation, scene alignment and OOM retry."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

import lead.inference.qwendrive_batch as batch_module


def test_left_padding_images_and_generated_rows_stay_aligned(monkeypatch):
    class Tokenizer:
        pad_token_id = 0
        def decode(self, ids, skip_special_tokens):
            return ','.join(str(x) for x in ids if not skip_special_tokens or x not in (0, 99))
    observed = {}
    def generate(**kwargs):
        observed.update(kwargs)
        return torch.cat([kwargs['input_ids'], torch.tensor([[71, 99, 0], [72, 73, 99]])], dim=1)
    model = SimpleNamespace(
        device='cpu', processor=SimpleNamespace(tokenizer=Tokenizer(), im_end_id=99),
        vlm=SimpleNamespace(generate=generate, generation_config=SimpleNamespace(eos_token_id=99)),
        _modality_ids=lambda ids: (ids == 50).long(),
    )
    monkeypatch.setattr(batch_module, 'decoding_defaults', lambda: {'seed': 3407, 'presence_penalty': 0.0})
    encoded = [
        {'input_ids': torch.tensor([[50, 11]]), 'pixel_values': torch.full((3, 2), 1.0),
         'image_grid_thw': torch.tensor([[1, 2, 2]] * 3)},
        {'input_ids': torch.tensor([[50, 50, 12]]), 'pixel_values': torch.full((3, 2), 2.0),
         'image_grid_thw': torch.tensor([[1, 4, 4]] * 3)},
    ]
    result = batch_module.generate_encoded_batch(model, encoded, 3)
    assert observed['input_ids'].tolist() == [[0, 50, 11], [50, 50, 12]]
    assert observed['attention_mask'].tolist() == [[0, 1, 1], [1, 1, 1]]
    assert observed['mm_token_type_ids'].tolist() == [[0, 1, 0], [1, 1, 0]]
    assert observed['pixel_values'][:, 0].tolist() == [1, 1, 1, 2, 2, 2]
    assert observed['image_grid_thw'].tolist() == [[1, 2, 2]] * 3 + [[1, 4, 4]] * 3
    assert [row['text'] for row in result] == ['71', '72,73']
    assert [row['tokens'] for row in result] == [2, 3]
    assert not any(row['hit_limit'] for row in result)


def test_two_batched_calls_keep_each_scenes_own_analysis(tmp_path, monkeypatch):
    prompts, calls = [], []
    class Processor:
        def encode_vqa(self, images, question, device):
            value = images[0].getpixel((0, 0))[0]
            prompts.append((value, question))
            return {'frame': value}
    def generate(model, encoded, max_new_tokens):
        stage = 'analysis' if not calls else 'answer'
        calls.append([entry['frame'] for entry in encoded])
        return [{'text': f'{stage}-{entry["frame"]}', 'raw': 'raw', 'tokens': 5,
                 'hit_limit': False} for entry in encoded]
    monkeypatch.setattr(batch_module, 'generate_encoded_batch', generate)
    entries = []
    for i in (10, 20):
        path = tmp_path / f'A/r/rgb/{i}.jpg'
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (1152, 384), (i, 0, 0)).save(path, format='PNG')
        entries.append({'image_key': f'A/r/rgb/{i}.jpg', 'navigation': 'follow_lane', 'speed': 3.0})
    entries.insert(1, {'image_key': 'A/r/rgb/missing.jpg', 'navigation': 'follow_lane', 'speed': 3.0})
    generator = batch_module.BatchIntentGenerator(SimpleNamespace(processor=Processor()), tmp_path)
    try:
        results = generator.generate(entries)
    finally:
        generator.close()
    assert calls == [[10, 20], [10, 20]]  # Two generate calls, not two calls per frame.
    assert results[0]['raw_text'] == 'answer-10'
    assert results[1]['error'] is not None
    assert results[2]['raw_text'] == 'answer-20'
    for frame, prompt in prompts:
        if 'fallible scene analysis' in prompt:
            assert f'analysis-{frame}' in prompt
            assert f'analysis-{30-frame}' not in prompt


def test_64_batch_oom_retries_without_writing_failed_frames(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('p7_generate_batch', 'scripts/p7/generate_p7_intents.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    manifest = tmp_path / 'manifest.jsonl'
    entries = [{'image_key': f'A/r/rgb/{i:04d}.jpg', 'navigation': 'follow_lane', 'speed': 3.0}
               for i in range(64)]
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in entries))
    calls = []
    class Generator:
        def __init__(self, *_args, **_kwargs): pass
        def generate(self, rows):
            calls.append(len(rows))
            if len(calls) == 1:
                raise torch.cuda.OutOfMemoryError('test')
            return [{'raw_text': '{}', 'analysis': '', 'error': None} for row in rows]
        def close(self): pass
    monkeypatch.setattr(batch_module, 'BatchIntentGenerator', Generator)
    monkeypatch.setattr(script, 'load_vqa_model', lambda *_: None)
    args = SimpleNamespace(model=str(model), manifest=manifest, output_dir=tmp_path / 'out',
                           data_root=tmp_path, device='cpu', shard=0, num_shards=1,
                           analysis_tokens=512, answer_tokens=384, retry_failed=False,
                           batch_size=64, preprocess_workers=4)
    script.generate(args)
    rows = [json.loads(line) for line in (args.output_dir / 'shard-00000.jsonl').read_text().splitlines()][1:]
    assert calls == [64, 32, 32]
    assert [row['image_key'] for row in rows] == [row['image_key'] for row in entries]
    assert all(row['error'] is None and row['generation_batch_size'] == 32 for row in rows)
