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
