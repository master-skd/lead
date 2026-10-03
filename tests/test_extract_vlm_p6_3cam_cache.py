import numpy as np

from scripts.p4.extract_vlm_p6_3cam import (
    atomic_save_cache,
    split_image_features,
    valid_cache_file,
)


def test_atomic_cache_save_and_resume_validation(tmp_path):
    path = tmp_path / "scenario" / "route" / "0001.npy"
    feature = np.ones((12, 36, 2560), dtype=np.float16)

    atomic_save_cache(str(path), feature)

    assert valid_cache_file(str(path))
    np.testing.assert_array_equal(np.load(path, allow_pickle=False), feature)
    assert list(path.parent.iterdir()) == [path]


def test_truncated_or_wrong_shape_cache_is_not_skipped(tmp_path):
    path = tmp_path / "0001.npy"
    atomic_save_cache(str(path), np.zeros((12, 36, 2560), dtype=np.float16))

    with path.open("r+b") as stream:
        stream.truncate(path.stat().st_size - 10)
    assert not valid_cache_file(str(path))

    atomic_save_cache(str(path), np.ones((12, 36, 2560), dtype=np.float16))
    assert valid_cache_file(str(path))

    np.save(path, np.zeros((12, 12, 2560), dtype=np.float16))
    assert not valid_cache_file(str(path))


def test_wrong_dtype_cache_is_not_skipped(tmp_path):
    path = tmp_path / "0001.npy"
    np.save(path, np.zeros((12, 36, 2560), dtype=np.float32))
    assert not valid_cache_file(str(path))


def test_split_batched_image_tokens_ignores_padding():
    image_token = 99
    ids = np.array([[1, 99, 99, 99, 99, 2], [0, 99, 99, 99, 99, 2]])
    hidden = np.arange(12, dtype=np.float16).reshape(2, 6, 1)
    grids = np.array([[1, 2, 2], [1, 2, 2]])

    features = split_image_features(hidden, ids, grids, image_token, merge=1)

    assert len(features) == 2
    np.testing.assert_array_equal(features[0][0].reshape(-1), np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(features[1][0].reshape(-1), np.array([7, 8, 9, 10]))
