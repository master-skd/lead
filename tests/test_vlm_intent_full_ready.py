import numpy as np
import pytest

from scripts.p4.check_vlm_intent_full_ready import completed_frames
from scripts.p4.precompute_lanegraph_intent import atomic_save_npy, valid_npy


def test_completed_frames_checks_resume_accounting(tmp_path):
    (tmp_path / "shard_0.log").write_text(
        "[shard 0/2] frames: 5\n[shard 0] finished: done=3 skipped=2 repaired=1 bad=0\n"
    )
    (tmp_path / "shard_1.log").write_text(
        "[shard 1/2] frames: 4\n[shard 1] finished: done=4 skipped=0 bad=0\n"
    )
    assert completed_frames(tmp_path, 2, "test") == 9

    (tmp_path / "shard_1.log").write_text(
        "[shard 1/2] frames: 4\n[shard 1] finished: done=3 skipped=0 bad=1\n"
    )
    with pytest.raises(ValueError, match="incomplete"):
        completed_frames(tmp_path, 2, "test")


def test_lanegraph_atomic_cache_validation(tmp_path):
    path = tmp_path / "scenario" / "route" / "0001.npy"
    feature = np.ones((1, 4, 5), dtype=np.float16)
    atomic_save_npy(str(path), feature)
    assert valid_npy(str(path), feature.shape)
    with path.open("r+b") as stream:
        stream.truncate(path.stat().st_size - 1)
    assert not valid_npy(str(path), feature.shape)
