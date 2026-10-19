"""Match the saved CARLA JPEG image boundary used to train visual intent."""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def carla_bgr_to_vlm_rgb(image_bgr: np.ndarray, jpeg_quality: int) -> np.ndarray:
    """Encode a stitched CARLA BGR strip like expert_data, then read it like training.

    The dataset writes the full BGR strip with ``cv2.imwrite`` and the intent
    loader reads that JPEG with ``PIL.Image.open(...).convert('RGB')``.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("CARLA image must be HxWx3 uint8 BGR")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    success, encoded = cv2.imencode(
        ".jpg", np.ascontiguousarray(image_bgr),
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not success:
        raise RuntimeError("failed to JPEG-encode CARLA image for VLM")
    with Image.open(BytesIO(encoded.tobytes())) as decoded:
        return np.asarray(decoded.convert("RGB")).copy()
