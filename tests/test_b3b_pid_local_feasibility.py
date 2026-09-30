import numpy as np

from scripts.p4.audit_b3b_pid_local_feasibility import hdmap_path, road_fraction


def test_hdmap_key_resolution(tmp_path):
    path = hdmap_path(tmp_path, "Accident__Town03_route_1__0008")
    assert path == tmp_path / "Accident/Town03_route_1/hdmap/0008.png"


def test_swept_road_fraction_detects_offroad():
    hdmap = np.ones((256, 256), dtype=np.uint8)
    positions = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    yaws = np.zeros(2, dtype=np.float32)
    assert road_fraction(hdmap, positions, yaws) == 1.0
    hdmap[:, 130:] = 0
    assert road_fraction(hdmap, positions, yaws) < 1.0
