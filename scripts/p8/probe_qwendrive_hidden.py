"""Measure actual hidden forward latency and token layout before training."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from PIL import Image
from lead.inference.qwendrive_hidden import QwenDriveHiddenExtractor


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--images', nargs='+', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--microbatch', type=int, default=8)
    ap.add_argument('--command', default='follow_lane')
    ap.add_argument('--speed', type=float, default=3.0)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--check-batch-parity', action='store_true')
    ap.add_argument('--output', type=Path, default=Path('outputs/p8_hidden_probe.json'))
    args = ap.parse_args()
    if args.repeats < 2:
        raise ValueError('at least two repetitions (first is warmup)')
    model = QwenDriveHiddenExtractor(args.model, args.device, args.microbatch)
    images = []
    for path in args.images:
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    rows = []
    if model.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(model.device)
    for step in range(args.repeats):
        start = time.monotonic()
        arrays, metrics = model.extract(images, [args.command]*len(images), [args.speed]*len(images))
        row = dict(step=step, seconds=time.monotonic()-start,
                   hidden_shape=list(arrays['qwen_hidden'].shape), **metrics)
        rows.append(row)
        print(json.dumps(row), flush=True)
    durations = sorted(row['seconds'] for row in rows[1:])
    report = dict(model=args.model, device=args.device, gpu=torch.cuda.get_device_name(model.device) if model.device.type=='cuda' else 'cpu',
                  batch=len(images), microbatch=args.microbatch, rows=rows,
                  p50_seconds=statistics.median(durations), p95_seconds=durations[min(len(durations)-1, int(len(durations)*0.95))],
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(model.device) if model.device.type=='cuda' else None)
    if args.check_batch_parity:
        import numpy as np
        comparisons = []
        for i, image in enumerate(images):
            single, _ = model.extract([image], [args.command], [args.speed])
            a = arrays['qwen_hidden'][i, arrays['qwen_token_mask'][i]].astype(np.float32)
            b = single['qwen_hidden'][0, single['qwen_token_mask'][0]].astype(np.float32)
            if a.shape != b.shape or not np.array_equal(arrays['qwen_view_ids'][i, arrays['qwen_token_mask'][i]],
                                                       single['qwen_view_ids'][0, single['qwen_token_mask'][0]]):
                raise ValueError('batched/single token layout mismatch')
            cosine = np.dot(a.ravel(), b.ravel()) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
            comparisons.append(dict(image=args.images[i], mae=float(abs(a-b).mean()),
                                    max_error=float(abs(a-b).max()), cosine=float(cosine)))
        report['batch_parity'] = comparisons
        print(json.dumps({'batch_parity': comparisons}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('p50_seconds','p95_seconds','peak_allocated_bytes')}))


if __name__ == '__main__':
    main()
