"""The planner-only control must use exactly the recorded P8 frames."""
import numpy as np
import pytest
from lead.data_loader.frame_subset import select_frame_manifest


def test_manifest_selects_exact_keys_independent_of_disk_and_order(tmp_path):
    manifest=tmp_path/'frames.txt'
    manifest.write_text('Scenario/Route/rgb/00001.jpg\nScenario/Route/rgb/00003.jpg\n')
    images=np.asarray([f'/another/disk/Scenario/Route/rgb/{i:05d}.jpg'.encode() for i in range(4)])
    indices=np.asarray([3,2,1,0])
    assert select_frame_manifest(images,indices,manifest).tolist()==[3,1]
    with pytest.raises(ValueError,match='absent'):
        select_frame_manifest(images,np.asarray([1,0]),manifest)
    manifest.write_text('Scenario/Route/rgb/00001.jpg\n'*2)
    with pytest.raises(ValueError,match='unique'):
        select_frame_manifest(images,indices,manifest)
