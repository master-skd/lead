"""Check the trained P8 feature contract before starting closed-loop extraction."""
import json
from pathlib import Path
from lead.tfv6.hidden_state_cache import qwen_feature_fingerprint


def verify_training_features(signature_path, model_dir):
    expected=json.loads(Path(signature_path).read_text())
    contract=dict(prompt_version='p8-hidden-v1',layer='final_normalized',transport_dtype='float16',
                  compute_dtype='bfloat16',view_order=['front','front_left','front_right','text'])
    mismatches=[key for key,value in contract.items() if expected.get(key)!=value]
    actual=qwen_feature_fingerprint(Path(model_dir))
    for key in ('model_config_sha256','model_weights','model_assets','qwen_sources','extractor_sha256'):
        if expected.get(key)!=actual[key]: mismatches.append(key)
    config=json.loads((Path(model_dir)/'config.json').read_text())
    if expected.get('hidden_dim')!=config['vlm_config']['text_config']['hidden_size']:
        mismatches.append('hidden_dim')
    # Transformers implements the forward and official image processor.
    for package in ('transformers','torchvision','pillow'):
        if expected['runtime'].get(package)!=actual['runtime'].get(package):
            mismatches.append(f'runtime.{package}')
    if mismatches:
        raise ValueError('P8 training/evaluation feature contract differs: '+', '.join(mismatches))
    warnings={package:dict(training=version,evaluation=actual['runtime'].get(package))
              for package,version in expected['runtime'].items()
              if version!=actual['runtime'].get(package)}
    return dict(verified=True,hidden_dim=expected['hidden_dim'],runtime_differences=warnings)
