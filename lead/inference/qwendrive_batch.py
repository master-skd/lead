"""Batch independent Qwen-Drive VQA conversations using the official processor."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torch

from lead.inference.qwendrive_intent_service import _views, intent_prompts


def decoding_defaults():
    from qwen_drive.modeling_qwen_drive import VQA_DECODE_DEFAULTS
    return dict(VQA_DECODE_DEFAULTS)


def generate_encoded_batch(model, encoded, max_new_tokens):
    """Left-pad text, concatenate images in conversation order, decode each row."""
    if not encoded:
        return []
    tokenizer = model.processor.tokenizer
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if not isinstance(pad_id, int):
        raise ValueError('Qwen-Drive batch generation requires a scalar pad/eos token ID')
    width = max(item['input_ids'].shape[1] for item in encoded)
    ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    for index, item in enumerate(encoded):
        length = item['input_ids'].shape[1]
        ids[index, -length:] = item['input_ids'][0]
        attention[index, -length:] = 1
    params = decoding_defaults()
    seed = params.pop('seed', None)
    if seed is not None:
        torch.manual_seed(seed)
    params.pop('presence_penalty', None)  # Official HF wrapper drops this too.
    params.update(max_new_tokens=max_new_tokens, pad_token_id=pad_id)
    ids = ids.to(model.device)
    with torch.inference_mode():
        output = model.vlm.generate(
            input_ids=ids, attention_mask=attention.to(model.device),
            pixel_values=torch.cat([item['pixel_values'] for item in encoded]).to(model.device),
            image_grid_thw=torch.cat([item['image_grid_thw'] for item in encoded]).to(model.device),
            mm_token_type_ids=model._modality_ids(ids), **params,
        )
    if output.shape[0] != len(encoded):
        raise ValueError('Qwen-Drive returned a different number of batch rows')
    eos = model.vlm.generation_config.eos_token_id
    terminators = set(eos if isinstance(eos, (list, tuple)) else [eos])
    terminators.add(model.processor.im_end_id)
    result = []
    for row in output[:, width:].cpu():
        tokens = row.tolist()
        stop = next((i + 1 for i, token in enumerate(tokens) if token in terminators), len(tokens))
        tokens = tokens[:stop]
        raw = tokenizer.decode(tokens, skip_special_tokens=False)
        text = tokenizer.decode(tokens, skip_special_tokens=True).split('</think>')[-1].strip()
        result.append({'text': text, 'raw': raw, 'tokens': len(tokens),
                       'hit_limit': bool(len(tokens) >= max_new_tokens and
                                         (not tokens or tokens[-1] not in terminators))})
    return result


class BatchIntentGenerator:
    """Two true batched HF generate calls per group, with CPU image workers."""
    def __init__(self, model, media_root, *, analysis_tokens=512, answer_tokens=384, workers=8):
        self.model = model
        self.processor = model.processor  # Resolve lazy processor before starting threads.
        self.media_root = media_root
        self.analysis_tokens = analysis_tokens
        self.answer_tokens = answer_tokens
        self.analysis_prompt, self.final_prompt = intent_prompts()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='p7-images')

    def _analysis_input(self, entry):
        image = str((self.media_root / entry['image_key']).resolve())
        images = _views(image, self.media_root)
        prompt = self.analysis_prompt(entry['navigation'], entry['speed'])
        return images, self.processor.encode_vqa(images, prompt, device='cpu')

    def _answer_input(self, entry, images, analysis):
        prompt = self.final_prompt(entry['navigation'], entry['speed'], analysis)
        return self.processor.encode_vqa(images, prompt, device='cpu')

    def generate(self, entries):
        results = [None] * len(entries)
        valid, images, encoded = [], [], []
        futures = [self.pool.submit(self._analysis_input, entry) for entry in entries]
        for index, future in enumerate(futures):
            try:
                views, inputs = future.result()
                valid.append(index)
                images.append(views)
                encoded.append(inputs)
            except Exception as exc:
                results[index] = {'raw_text': '', 'analysis': '',
                                  'error': f'{type(exc).__name__}: {exc}'}
        analysis = generate_encoded_batch(self.model, encoded, self.analysis_tokens)
        futures = [self.pool.submit(self._answer_input, entries[index], views, item['text'])
                   for index, views, item in zip(valid, images, analysis, strict=True)]
        answer_indices, answer_inputs, answer_analysis = [], [], []
        for index, item, future in zip(valid, analysis, futures, strict=True):
            try:
                answer_inputs.append(future.result())
                answer_indices.append(index)
                answer_analysis.append(item)
            except Exception as exc:
                results[index] = {'raw_text': '', 'analysis': item['text'],
                                  'error': f'{type(exc).__name__}: {exc}'}
        answers = generate_encoded_batch(self.model, answer_inputs, self.answer_tokens)
        for index, item, answer in zip(answer_indices, answer_analysis, answers, strict=True):
            results[index] = {
                'raw_text': answer['text'], 'analysis': item['text'], 'error': None,
                'analysis_raw': item['raw'], 'analysis_tokens': item['tokens'],
                'analysis_hit_limit': item['hit_limit'], 'answer_tokens': answer['tokens'],
                'answer_hit_limit': answer['hit_limit'],
            }
        return results

    def close(self):
        self.pool.shutdown(wait=True)
