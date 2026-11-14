"""Stable frame sampling independent of epoch, bucket order and DDP rank."""
import hashlib
import math
import os
import numpy as np


def select_frame_fraction(images, indices, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError('frame fraction must be in (0,1]')
    from lead.tfv6.semantic_intent_cache import image_key
    # Rank unique frame keys once; repeated bucket entries must not bias selection.
    keys = [image_key(os.fsdecode(images[i])) for i in indices]
    unique = set(keys)
    count = max(1, math.floor(len(unique) * fraction)) if unique else 0
    selected = set(sorted(unique, key=lambda key: (
        hashlib.sha256(f'{seed}:{key}'.encode()).digest(), key))[:count])
    return np.asarray([i for i, key in zip(indices, keys) if key in selected], dtype=indices.dtype)


def select_frame_manifest(images, indices, manifest):
    """Select exactly the recorded frame keys; reject missing/duplicate entries."""
    from pathlib import Path
    from lead.tfv6.semantic_intent_cache import image_key
    lines = Path(manifest).read_text().splitlines()
    keys = {line.strip() for line in lines if line.strip()}
    if not keys or len(keys) != len(lines):
        raise ValueError('Frame manifest must contain unique, non-empty image keys')
    selected = [i for i in indices if image_key(os.fsdecode(images[i])) in keys]
    found = {image_key(os.fsdecode(images[i])) for i in selected}
    if found != keys:
        raise ValueError(f'{len(keys - found)} manifest frames absent from active posttrain pool')
    return np.asarray(selected, dtype=indices.dtype)
