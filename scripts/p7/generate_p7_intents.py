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
    if not 0 <= args.shard < args.num_shards or min(args.analysis_tokens, args.answer_tokens) < 1:
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
    histogram = Counter()
    with output.open('a') as stream:
        if not header_seen:
            stream.write(json.dumps({'type': 'metadata', 'signature': expected_signature,
                                     'model': str(Path(args.model).resolve())}) + '\n')
            stream.flush()
        for index, entry in enumerate(pending, 1):
            image = (media_root / entry['image_key']).resolve()
            started = time.monotonic()
            try:
                result = generate_intent(
                    model, str(image), entry['navigation'], entry['speed'], media_root,
                    analysis_tokens=args.analysis_tokens, answer_tokens=args.answer_tokens,
                )
            except Exception as exc:
                result = {'raw_text': '', 'analysis': '', 'error': f'{type(exc).__name__}: {exc}'}
            proposal = parse_service_result(result)
            packed = prepare_planner_intents(proposal, entry['navigation'])
            count = sum(packed['valid'])
            histogram[count] += 1
            row = {**entry, **result, 'image': str(image), 'parsed': proposal,
                   'valid_branches': count, 'fallback_required': packed['fallback_required'],
                   'rejected': packed['rejected'], 'seconds': time.monotonic() - started}
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            if index % 10 == 0 or index == len(pending):
                os.fsync(stream.fileno())
                print(f'shard={args.shard} {index}/{len(pending)} '
                      f'valid_histogram={dict(histogram)} last={row["seconds"]:.1f}s', flush=True)


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
