"""Export the same fixed tenth of posttrain frames used by online P8 training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--posttrain-config',default='outputs/local_training/posttrain/config.json')
    ap.add_argument('--output',type=Path,default=Path('data/p8/hidden_manifest.jsonl'))
    ap.add_argument('--fraction',type=float,default=.1)
    ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args()
    from lead.training.config_training import TrainingConfig
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.frame_subset import select_frame_fraction
    from lead.common.common_utils import read_pickle
    from lead.data_loader.carla_dataset_utils import command_to_one_hot
    from lead.inference.qwendrive_intent_client import NAV_COMMANDS
    from lead.tfv6.semantic_intent_cache import image_key
    config=TrainingConfig(json.loads(Path(args.posttrain_config).read_text()))
    config.use_vlm_intent=config.online_joint_training=config.use_semantic_intent_planner=False
    config.use_hidden_intent_planner=True
    config.hidden_intent_cache=None
    config.use_persistent_cache=config.use_training_session_cache=False
    dataset=CARLAData(config.carla_data,config,random=False,build_buckets=True)
    indices=select_frame_fraction(dataset.images,np.arange(len(dataset.images)),args.fraction,args.seed)
    selected={}
    for i in indices:
        key=image_key(os.fsdecode(dataset.images[i]))
        selected.setdefault(key,i)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.tmp')
    with temporary.open('w') as stream:
        for count,key in enumerate(sorted(selected),1):
            i=selected[key]
            meta=read_pickle(os.fsdecode(dataset.metas[i]))
            suffix='_gps' if config.use_noisy_tp else ''
            commands=meta[f'next{suffix}_commands_{config.tp_pop_distance}']
            command=NAV_COMMANDS[int(command_to_one_hot(commands[0]).argmax())]
            stream.write(json.dumps(dict(image_key=key,navigation=command,speed=float(meta['speed'])))+'\n')
            if count%10000==0:
                print(f'exported {count}/{len(selected)} selected frames',flush=True)
    os.replace(temporary,args.output)
    args.output.with_suffix('.meta.json').write_text(json.dumps(dict(fraction=args.fraction,seed=args.seed,
        frames=len(selected),posttrain_config_sha256=hashlib.sha256(Path(args.posttrain_config).read_bytes()).hexdigest()),indent=2)+'\n')
    print(f'Exported {len(selected)}/{len(dataset.images)} frames -> {args.output}')


if __name__=='__main__':
    main()
