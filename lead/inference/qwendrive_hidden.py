"""Frozen Qwen-Drive final-layer tokens from ONE forward, without LM logits."""
import time
import numpy as np
import torch
from lead.inference.qwendrive_intent_client import NAV_COMMANDS
from lead.inference.qwendrive_intent_service import split_views, load_vqa_model

PROMPT_VERSION = 'p8-hidden-v1'


def hidden_prompt(command, speed):
    if command not in NAV_COMMANDS:
        raise ValueError(f'unknown Qwen-Drive navigation: {command!r}')
    # CARLA records contain signed velocities, including small negatives around
    # standstill. Preserve the dataset value in train/cache/live inference.
    if not np.isfinite(speed):
        raise ValueError(f'non-finite Qwen-Drive speed: {speed!r}')
    return ('Understand the driving scene for route planning.\n'
            f'Navigation: {command}. Ego speed: {speed:.3f} m/s.')


def pack_inputs(processor, strips, commands, speeds):
    encoded = [processor.encode_vqa(split_views(image), hidden_prompt(command, speed), device='cpu')
               for image, command, speed in zip(strips, commands, speeds, strict=True)]
    if not encoded:
        raise ValueError('empty hidden-state batch')
    lengths = [item['input_ids'].shape[1] for item in encoded]
    ids = torch.full((len(encoded), max(lengths)), processor.tokenizer.pad_token_id or 0, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    views = torch.full_like(ids, -1)
    for b, item in enumerate(encoded):
        tokens = item['input_ids'][0]
        offset = ids.shape[1] - len(tokens)  # left padding, consistent for train/eval
        ids[b, offset:] = tokens
        mask[b, offset:] = True
        views[b, offset:] = 3
        positions = (tokens == processor.image_token_id).nonzero().flatten()
        grid = item['image_grid_thw']
        counts = (grid.prod(-1) // processor.merge_size ** 2).tolist()
        if len(counts) != 3 or sum(counts) != len(positions):
            raise ValueError('official Qwen image token/grid layout mismatch')
        at = 0
        for view, count in enumerate(counts):
            views[b, offset + positions[at:at+count]] = view
            at += count
    inputs = dict(input_ids=ids, attention_mask=mask.long(),
                  pixel_values=torch.cat([item['pixel_values'] for item in encoded]),
                  image_grid_thw=torch.cat([item['image_grid_thw'] for item in encoded]))
    return inputs, mask, views


class QwenDriveHiddenExtractor:
    def __init__(self, model_dir, device='cuda:0', microbatch=8):
        if microbatch < 1:
            raise ValueError('microbatch must be positive')
        self.model = load_vqa_model(model_dir, device)
        self.model.eval().requires_grad_(False)
        self.microbatch = microbatch
        self.device = torch.device(device)
        self.hidden_dim = self.model.config.vlm_config.text_config.hidden_size

    @torch.inference_mode()
    def extract(self, strips, commands, speeds):
        if not (len(strips) == len(commands) == len(speeds)):
            raise ValueError('hidden-state batch lengths differ')
        rows, views = [], []
        preprocess_seconds = forward_seconds = 0.0
        for start in range(0, len(strips), self.microbatch):
            end = start + self.microbatch
            tick = time.monotonic()
            inputs, mask, view = pack_inputs(self.model.processor, strips[start:end], commands[start:end], speeds[start:end])
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            inputs['mm_token_type_ids'] = self.model._modality_ids(inputs['input_ids'])
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            preprocess_seconds += time.monotonic() - tick
            tick = time.monotonic()
            # Bypass the generation wrapper AND vocabulary projection. Final
            # normalized image/text states, not autoregressive reasoning tokens.
            result = self.model.vlm.model(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
            hidden = result.last_hidden_state
            if hidden.shape[:2] != mask.shape or hidden.shape[-1] != self.hidden_dim:
                raise ValueError('unexpected Qwen-Drive hidden layout')
            if not torch.isfinite(hidden).all():
                raise ValueError('non-finite Qwen-Drive forward')
            hidden = hidden.float().cpu().numpy().astype(np.float16)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            forward_seconds += time.monotonic() - tick
            for b in range(len(mask)):
                valid = mask[b].numpy()
                rows.append(hidden[b, valid])
                views.append(view[b].numpy()[valid])
            del result, inputs
        if not rows:
            raise ValueError('empty hidden-state request')
        length = max(len(row) for row in rows)
        features = np.zeros((len(rows), length, self.hidden_dim), dtype=np.float16)
        token_mask = np.zeros((len(rows), length), dtype=bool)
        view_ids = np.full((len(rows), length), -1, dtype=np.int8)
        for b, row in enumerate(rows):
            features[b, :len(row)], token_mask[b, :len(row)], view_ids[b, :len(row)] = row, True, views[b]
        return dict(qwen_hidden=features, qwen_token_mask=token_mask, qwen_view_ids=view_ids), {
            'preprocess_seconds': preprocess_seconds, 'forward_seconds': forward_seconds,
            'prompt_version': PROMPT_VERSION, 'layer': 'final_normalized',
            'hidden_dim': self.hidden_dim, 'tokens': token_mask.sum(1).tolist(),
            'view_order': ['front', 'front_left', 'front_right', 'text'],
        }
