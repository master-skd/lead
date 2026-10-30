"""Generate resumable P7 JSONL shards, then index them for planner-only training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from lead.inference.qwendrive_intent_service import (
    generate_intent, intent_prompts, load_vqa_model,
)
from lead.tfv6.semantic_intent_cache import build_index, recover_jsonl
from lead.tfv6.semantic_intent_slots import parse_service_result, prepare_planner_intents


def signature(args):
    analysis_prompt, final_prompt = intent_prompts()
    model = Path(args.model)
    # Prompt contents and generation settings are fixed for all resumptions/shards.
    return {
        'schema_version': 1,
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'prompt_sha256': hashlib.sha256((analysis_prompt('follow_lane', 0.0) +
                                        final_prompt('follow_lane', 0.0, '')).encode()).hexdigest(),
        'model_config_sha256': hashlib.sha256((model / 'config.json').read_bytes()).hexdigest(),
        'weights': {p.name: p.stat().st_size for p in sorted(model.glob('*.safetensors'))},
        'analysis_tokens': args.analysis_tokens, 'answer_tokens': args.answer_tokens,
        'num_shards': args.num_shards,
    }


def generate(args):
    batch_size = getattr(args, 'batch_size', 1)
    if (not 0 <= args.shard < args.num_shards
            or min(args.analysis_tokens, args.answer_tokens, batch_size) < 1):
        raise ValueError('invalid shard or token settings')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f'shard-{args.shard:05d}.jsonl'
    expected_signature = signature(args)
    completed = {}
    header_seen = False
    for index, row in enumerate(recover_jsonl(output)):
        if index == 0:
            if row.get('type') != 'metadata' or row['signature'] != expected_signature:
                raise ValueError(f'cannot resume {output}: generation settings changed')
            header_seen = True
        else:
            completed[row['image_key']] = row.get('valid_branches', 0) > 0
    pending = []
    with args.manifest.open() as stream:
        for index, line in enumerate(stream):
            if index % args.num_shards != args.shard:
                continue
            row = json.loads(line)
            done = completed.get(row['image_key'])
            if done is not None and (done or not args.retry_failed):
                continue
            pending.append(row)
    print(f'shard={args.shard}: completed={len(completed)}, pending={len(pending)}', flush=True)
    if not pending:
        return
    media_root = args.data_root.resolve(strict=True)
    model = load_vqa_model(args.model, args.device)
    import torch
    batcher = None
    if batch_size > 1:
        from lead.inference.qwendrive_batch import BatchIntentGenerator
        batcher = BatchIntentGenerator(
            model, media_root, analysis_tokens=args.analysis_tokens,
            answer_tokens=args.answer_tokens, workers=getattr(args, 'preprocess_workers', 8),
        )
    print(f'shard={args.shard}: target_batch_size={batch_size}', flush=True)
    histogram = Counter()
    with output.open('a') as stream:
        if not header_seen:
            stream.write(json.dumps({'type': 'metadata', 'signature': expected_signature,
                                     'model': str(Path(args.model).resolve())}) + '\n')
            stream.flush()
        offset = 0
        while offset < len(pending):
            entries = pending[offset:offset + batch_size]
            measure_cuda = str(args.device).startswith('cuda') and torch.cuda.is_available()
            if measure_cuda:
                torch.cuda.reset_peak_memory_stats(args.device)
            started = time.monotonic()
            try:
                if batcher is not None:
                    results = batcher.generate(entries)
                else:
                    entry = entries[0]
                    try:
                        results = [generate_intent(
                            model, str((media_root / entry['image_key']).resolve()),
                            entry['navigation'], entry['speed'], media_root,
                            analysis_tokens=args.analysis_tokens, answer_tokens=args.answer_tokens,
                        )]
                    except torch.cuda.OutOfMemoryError:
                        raise
                    except Exception as exc:
                        results = [{'raw_text': '', 'analysis': '',
                                    'error': f'{type(exc).__name__}: {exc}'}]
            except torch.cuda.OutOfMemoryError as exc:
                if batch_size <= 1:
                    raise
                import gc
                import traceback
                traceback.clear_frames(exc.__traceback__)
                gc.collect()
                torch.cuda.empty_cache()
                batch_size = max(1, len(entries) // 2)
                print(f'shard={args.shard}: CUDA OOM; reducing batch_size to {batch_size}; '
                      'retrying unwritten frames', flush=True)
                continue
            elapsed = time.monotonic() - started
            peak_gib = (torch.cuda.max_memory_allocated(args.device) / 1024**3
                        if measure_cuda else None)
            for entry, result in zip(entries, results, strict=True):
                proposal = parse_service_result(result)
                packed = prepare_planner_intents(proposal, entry['navigation'])
                count = sum(packed['valid'])
                histogram[count] += 1
                row = {**entry, **result,
                       'image': str((media_root / entry['image_key']).resolve()), 'parsed': proposal,
                       'valid_branches': count, 'fallback_required': packed['fallback_required'],
                       'rejected': packed['rejected'], 'seconds': elapsed / len(entries),
                       'batch_seconds': elapsed, 'generation_batch_size': len(entries),
                       'batch_peak_allocated_gib': peak_gib,
                       'generation_backend': 'hf_batch' if batcher else 'hf_serial'}
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            offset += len(entries)
            print(f'shard={args.shard} {offset}/{len(pending)} batch={len(entries)} '
                  f'batch_seconds={elapsed:.1f} frames_per_second={len(entries)/max(elapsed, 1e-9):.3f} '
                  f'peak_allocated_gib={peak_gib} '
                  f'valid_histogram={dict(histogram)}', flush=True)
    if batcher is not None:
        batcher.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest='action', required=True)
    gen = commands.add_parser('generate')
    gen.add_argument('--manifest', type=Path, required=True)
    gen.add_argument('--output-dir', type=Path, required=True)
    gen.add_argument('--model', required=True)
    gen.add_argument('--data-root', type=Path, default=Path('data/carla_leaderboard2/data'))
    gen.add_argument('--num-shards', type=int, default=1)
    gen.add_argument('--shard', type=int, default=0)
    gen.add_argument('--device', default='cuda:0')
    gen.add_argument('--analysis-tokens', type=int, default=512)
    gen.add_argument('--answer-tokens', type=int, default=384)
    gen.add_argument('--batch-size', type=int, default=8)
    gen.add_argument('--preprocess-workers', type=int, default=8)
    gen.add_argument('--retry-failed', action='store_true')
    idx = commands.add_parser('index')
    idx.add_argument('--manifest', type=Path, required=True)
    idx.add_argument('--shard-dir', type=Path, required=True)
    idx.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.action == 'generate':
        generate(args)
    else:
        print(json.dumps(build_index(args.manifest, args.shard_dir, args.output), indent=2))


if __name__ == '__main__':
    main()
