"""Prepare, generate resumable packed feature shards, and finalize a complete index."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from PIL import Image
from lead.tfv6.hidden_state_cache import (cache_metadata, initialize_cache, manifest_rows,
                                        CacheShardWriter, build_cache_index)


def generate(args):
    signature=cache_metadata(args.cache_dir)
    # Re-check the original manifest/model fingerprints before reusing any shard.
    initialize_cache(args.cache_dir,args.cache_dir/'manifest.jsonl',args.model,signature['num_shards'])
    if not 0<=args.shard<signature['num_shards'] or args.batch_size<1 or args.workers<1:
        raise ValueError('invalid shard/batch/workers')
    writer=CacheShardWriter(args.cache_dir,args.shard,signature)
    try:
        entries=[row for i,row in enumerate(manifest_rows(args.cache_dir/'manifest.jsonl'))
                 if i%signature['num_shards']==args.shard and row['image_key'] not in writer.completed]
        print(f'shard={args.shard} completed={len(writer.completed)} pending={len(entries)}',flush=True)
        if not entries:
            return
        import gc
        import traceback
        import torch
        from lead.inference.qwendrive_hidden import QwenDriveHiddenExtractor
        extractor=QwenDriveHiddenExtractor(args.model,args.device,args.batch_size)
        if str(extractor.model.vlm.dtype).removeprefix('torch.')!=signature['compute_dtype']:
            raise ValueError('Qwen compute dtype differs from prepared cache signature')
        if extractor.hidden_dim!=signature['hidden_dim']:
            raise ValueError('actual Qwen hidden dimension differs from cache signature')
        root=args.data_root.resolve(strict=True)
        def load(row):
            path=(root/row['image_key']).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError('cache image outside data root')
            with Image.open(path) as image:
                return image.convert('RGB')
        batch_size=args.batch_size
        offset=0
        print(f'shard={args.shard} batch_size={batch_size} workers={args.workers}',flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while offset<len(entries):
                batch=entries[offset:offset+batch_size]
                start=time.monotonic()
                images=list(pool.map(load,batch))
                try:
                    arrays,metrics=extractor.extract(images,[row['navigation'] for row in batch],[row['speed'] for row in batch])
                except torch.cuda.OutOfMemoryError as exc:
                    if len(batch)<=1:
                        raise
                    batch_size=max(1,len(batch)//2)
                    extractor.microbatch=batch_size
                    traceback.clear_frames(exc.__traceback__)
                    del images
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f'shard={args.shard} CUDA OOM: reducing batch to {batch_size}; '
                          'retrying the same unwritten frames',flush=True)
                    continue
                payloads=[]
                for i,row in enumerate(batch):
                    valid=arrays['qwen_token_mask'][i]
                    payloads.append((row,arrays['qwen_hidden'][i,valid],arrays['qwen_view_ids'][i,valid]))
                writer.append_batch(payloads)
                elapsed=time.monotonic()-start
                print(json.dumps(dict(shard=args.shard,done=offset+len(batch),pending_total=len(entries),
                                      extraction_batch_size=len(batch),batch_seconds=elapsed,
                                      frames_per_second=len(batch)/elapsed,**metrics)),flush=True)
                offset+=len(batch)
                del arrays,payloads,images
    finally:
        writer.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='action',required=True)
    prepare=sub.add_parser('prepare')
    prepare.add_argument('--cache-dir',type=Path,required=True)
    prepare.add_argument('--manifest',type=Path,required=True)
    prepare.add_argument('--model',type=Path,required=True)
    prepare.add_argument('--shards',type=int,default=8)
    gen=sub.add_parser('generate')
    gen.add_argument('--cache-dir',type=Path,required=True)
    gen.add_argument('--model',type=Path,required=True)
    gen.add_argument('--data-root',type=Path,default=Path('data/carla_leaderboard2/data'))
    gen.add_argument('--shard',type=int,required=True)
    gen.add_argument('--device',default='cuda:0')
    gen.add_argument('--batch-size',type=int,default=32)
    gen.add_argument('--workers',type=int,default=16)
    idx=sub.add_parser('index')
    idx.add_argument('--cache-dir',type=Path,required=True)
    args=ap.parse_args()
    if args.action=='prepare':
        print(json.dumps(initialize_cache(args.cache_dir,args.manifest,args.model,args.shards),indent=2))
    elif args.action=='generate':
        generate(args)
    else:
        print(json.dumps(build_cache_index(args.cache_dir),indent=2))


if __name__=='__main__':
    main()
