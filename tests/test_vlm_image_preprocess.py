import cv2
import numpy as np
import pytest
from PIL import Image

from lead.inference.vlm_image_preprocess import carla_bgr_to_vlm_rgb


def test_carla_bgr_to_vlm_rgb_matches_saved_jpeg(tmp_path):
    rng = np.random.default_rng(19)
    bgr = rng.integers(0, 256, size=(32, 96, 3), dtype=np.uint8)
    quality = 30
    saved = tmp_path / "rgb.jpg"
    assert cv2.imwrite(
        str(saved), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )

    with Image.open(saved) as source:
        expected = np.asarray(source.convert("RGB"))
    actual = carla_bgr_to_vlm_rgb(bgr, quality)
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("quality", [0, 101])
def test_carla_bgr_to_vlm_rgb_rejects_invalid_quality(quality):
    with pytest.raises(ValueError, match="jpeg_quality"):
        carla_bgr_to_vlm_rgb(np.zeros((4, 4, 3), dtype=np.uint8), quality)
