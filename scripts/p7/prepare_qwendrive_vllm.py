"""Export a standalone Qwen3.5 VLM without changing source tensor bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct


def read_header(path):
    with Path(path).open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(size))
    return header, 8 + size


def tensor_plan(source):
    plan, names = [], set()
    for path in sorted(source.glob('*.safetensors')):
        header, start = read_header(path)
        for name, info in header.items():
            if name == '__metadata__' or not name.startswith('vlm.'):
                continue
            target = name.removeprefix('vlm.')
            if target in names:
                raise ValueError(f'duplicate VLM tensor: {target}')
            names.add(target)
            plan.append((path, start, target, info))
    if not plan or not any(name.startswith('model.visual.') for name in names):
        raise ValueError('no Qwen-Drive VLM/vision tensors found')
    plan.sort(key=lambda row: (str(row[0]), row[3]['data_offsets'][0]))
    return plan


def export(source, output, inspect_only=False):
    source = source.resolve(strict=True)
    original = json.loads((source / 'config.json').read_text())
    if original.get('model_type') != 'qwen_drive' or original.get('vlm_config', {}).get('model_type') != 'qwen3_5':
        raise ValueError('expected a dense Qwen-Drive/Qwen3.5 checkpoint')
    plan = tensor_plan(source)
    summary = {'tensors': len(plan), 'tensor_bytes': sum(row[3]['data_offsets'][1] - row[3]['data_offsets'][0] for row in plan)}
    print(json.dumps(summary), flush=True)
    if inspect_only:
        return summary
    if output.exists():
        raise FileExistsError(f'{output} already exists; use a new adapter directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.preparing')
    temporary.mkdir()  # Never silently reuse an interrupted/inconsistent export.
    try:
        header, offset = {'__metadata__': {'format': 'pt'}}, 0
        for _, _, name, info in plan:
            count = info['data_offsets'][1] - info['data_offsets'][0]
            header[name] = {**info, 'data_offsets': [offset, offset + count]}
            offset += count
        encoded = json.dumps(header, separators=(',', ':')).encode()
        encoded += b' ' * (-len(encoded) % 8)
        digest = hashlib.sha256()
        with (temporary / 'model.safetensors').open('wb') as dest:
            dest.write(struct.pack('<Q', len(encoded)))
            dest.write(encoded)
            current, stream = None, None
            try:
                for path, start, _, info in plan:
                    if path != current:
                        if stream is not None:
                            stream.close()
                        current, stream = path, path.open('rb')
                    begin, end = info['data_offsets']
                    stream.seek(start + begin)
                    remaining = end - begin
                    while remaining:
                        chunk = stream.read(min(16 * 1024**2, remaining))
                        if not chunk:
                            raise EOFError(f'truncated source tensor in {path}')
                        digest.update(chunk)
                        dest.write(chunk)
                        remaining -= len(chunk)
            finally:
                if stream is not None:
                    stream.close()
        config = dict(original['vlm_config'])
        config.update(architectures=['Qwen3_5ForConditionalGeneration'], dtype='bfloat16',
                      p7_qwendrive_config=original)
        config.pop('auto_map', None)
        (temporary / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        for name in ('tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json',
                     'processor_config.json', 'generation_config.json', 'special_tokens_map.json',
                     'vocab.json', 'merges.txt', 'chat_template.jinja', 'chat_template.json'):
            if (source / name).is_file():
                shutil.copyfile(source / name, temporary / name)
        summary.update(source=str(source), config_sha256=hashlib.sha256((source / 'config.json').read_bytes()).hexdigest(),
                       exported_payload_sha256=digest.hexdigest(), tensor_values_changed=False)
        (temporary / 'p7_vllm_export.json').write_text(json.dumps(summary, indent=2) + '\n')
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(f'Exported standalone VLM to {output}; source checkpoint unchanged', flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inspect-only', action='store_true')
    args = ap.parse_args()
    export(args.source, args.output, args.inspect_only)


if __name__ == '__main__':
    main()
