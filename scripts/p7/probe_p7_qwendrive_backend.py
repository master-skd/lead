"""Compare actual single-frame HF/vLLM latency and structured intent semantics."""
import argparse
import json
from pathlib import Path
import statistics
from types import SimpleNamespace
import time

from lead.inference.qwendrive_intent_service import _views, generate_intent_views, load_vqa_model
from lead.tfv6.semantic_intent_slots import parse_service_result, prepare_planner_intents


def compare(hf_path, vllm_path):
    hf = {row['image']: row for row in map(json.loads, hf_path.read_text().splitlines())}
    engine = {row['image']: row for row in map(json.loads, vllm_path.read_text().splitlines())}
    if set(hf) != set(engine) or not hf:
        raise ValueError('probe image sets must be identical and nonempty')
    matching = 0
    for image, a in hf.items():
        b = engine[image]
        if (a['navigation'], a['speed']) != (b['navigation'], b['speed']):
            raise ValueError(f'probe navigation/speed mismatch: {image}')
        pa = prepare_planner_intents(parse_service_result(a), a['navigation'])
        pb = prepare_planner_intents(parse_service_result(b), b['navigation'])
        keys = ('slots', 'scene_rules', 'valid', 'fallback_required')
        same = all(pa[key] == pb[key] for key in keys) and not pa['fallback_required']
        matching += same
        print(json.dumps({'image': image, 'hf_seconds': a['seconds'], 'vllm_seconds': b['seconds'],
                          'semantic_equal_and_valid': same, 'hf_fallback': pa['fallback_required'],
                          'vllm_fallback': pb['fallback_required']}))
    summary = {'samples': len(hf), 'semantic_equal_and_valid': matching,
               'hf_mean_seconds': statistics.mean(row['seconds'] for row in hf.values()),
               'vllm_mean_seconds': statistics.mean(row['seconds'] for row in engine.values())}
    print(json.dumps(summary, indent=2))
    return summary


class MeasuredHF:
    def __init__(self, model):
        self.model = model
        self.last_metrics = None

    def generate_text(self, images, question, *, max_new_tokens):
        from probe_qwendrive_branch_set_cpu import generate_text_with_metadata
        start = time.monotonic()
        text, metrics = generate_text_with_metadata(self.model, images, question, max_new_tokens)
        self.last_metrics = {**metrics, 'backend': 'hf', 'seconds': time.monotonic() - start}
        return SimpleNamespace(text=text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest='action', required=True)
    run = commands.add_parser('run')
    run.add_argument('--backend', choices=['hf', 'vllm'], required=True)
    run.add_argument('--model', required=True)
    run.add_argument('--images', type=Path, nargs='+', required=True)
    run.add_argument('--command', default='follow_lane')
    run.add_argument('--speed', type=float, default=3.0)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--analysis-tokens', type=int, default=512)
    run.add_argument('--answer-tokens', type=int, default=384)
    run.add_argument('--gpu-memory-utilization', type=float, default=0.5)
    run.add_argument('--max-model-len', type=int, default=4096)
    run.add_argument('--enforce-eager', action='store_true')
    diff = commands.add_parser('compare')
    diff.add_argument('--hf', type=Path, required=True)
    diff.add_argument('--vllm', type=Path, required=True)
    args = ap.parse_args()
    if args.action == 'compare':
        compare(args.hf, args.vllm)
        return
    start = time.monotonic()
    if args.backend == 'vllm':
        from lead.inference.qwendrive_vllm import VLLMQwenDrive
        model = VLLMQwenDrive(args.model, gpu_memory_utilization=args.gpu_memory_utilization,
                             max_model_len=args.max_model_len, enforce_eager=args.enforce_eager)
    else:
        model = MeasuredHF(load_vqa_model(args.model, 'cuda:0'))
    print(f'{args.backend} load_seconds={time.monotonic()-start:.3f}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as stream:
        for image in args.images:
            image = image.resolve(strict=True)
            views = _views(str(image), image.parent)
            start = time.monotonic()
            row = generate_intent_views(model, views, args.command, args.speed,
                                        analysis_tokens=args.analysis_tokens, answer_tokens=args.answer_tokens)
            row.update(image=str(image), navigation=args.command, speed=args.speed,
                       backend=args.backend, seconds=time.monotonic()-start)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            print(f'{image.name}: seconds={row["seconds"]:.3f} '
                  f'analysis={row["analysis_metrics"]} answer={row["answer_metrics"]}', flush=True)


if __name__ == '__main__':
    main()
