import numpy as np
import pytest

from scripts.p4.check_p6_intent_capture import compare_arrays, describe_intent


def test_compare_arrays_detects_identical_replay_and_shape_mismatch():
    reference = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float16)
    assert compare_arrays(reference, reference.copy()) == pytest.approx((0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_arrays(reference, reference[0])


def test_describe_intent_reports_empty_binary_mask():
    stats = describe_intent(np.full((320, 384), -10.0, dtype=np.float32))
    assert stats["coverage_0.50"] == 0.0
    assert stats["valid_anchors_0.50"] == 0
