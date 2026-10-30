"""Export the exact posttrain bucket frames and navigation inputs, without sensors."""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--posttrain-config', default='outputs/local_training/posttrain/config.json')
    ap.add_argument('--output', default='data/p7/intent_manifest.jsonl')
    ap.add_argument('--max-frames', type=int, default=0)
    args = ap.parse_args()
    from lead.training.config_training import TrainingConfig
    from lead.data_loader.carla_dataset import CARLAData
    from lead.common.common_utils import read_pickle
    from lead.data_loader.carla_dataset_utils import command_to_one_hot
    from lead.inference.qwendrive_intent_client import NAV_COMMANDS
    from lead.tfv6.semantic_intent_cache import image_key

    config = TrainingConfig(json.loads(Path(args.posttrain_config).read_text()),
                            raise_error_on_missing_key=False)
    config.use_vlm_intent = config.online_joint_training = False
    config.use_semantic_intent_planner = False
    config.use_persistent_cache = config.use_training_session_cache = False
    dataset = CARLAData(config.carla_data, config, random=False, build_buckets=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    count = 0
    with temporary.open('w') as stream:
        for image, metadata in zip(dataset.images, dataset.metas, strict=True):
            image = os.fsdecode(image)
            meta = read_pickle(os.fsdecode(metadata))
            suffix = '_gps' if config.use_noisy_tp else ''
            # Dataset's duplicate-TP removal always preserves the first entry.
            commands = meta[f'next{suffix}_commands_{config.tp_pop_distance}']
            command = NAV_COMMANDS[int(command_to_one_hot(commands[0]).argmax())]
            stream.write(json.dumps({'image_key': image_key(image),
                                     'image': str(Path(image).resolve()),
                                     'navigation': command, 'speed': float(meta['speed'])}) + '\n')
            count += 1
            if count % 10000 == 0:
                print(f'exported {count} frames', flush=True)
            if args.max_frames > 0 and count >= args.max_frames:
                break
    os.replace(temporary, output)
    print(f'Exported {count} frames to {output}', flush=True)


if __name__ == '__main__':
    main()
