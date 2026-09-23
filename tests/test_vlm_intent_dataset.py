import json
from types import SimpleNamespace

import numpy as np

from lead.data_loader.vlm_intent_dataset import VLMIntentDataset


class _FakeCarlaDataset:
    def __init__(self):
        self.config = SimpleNamespace()
        self.images = [
            f"/data/Scenario/Route/rgb/{frame:04d}.jpg".encode() for frame in range(3)
        ]

    def __getitem__(self, index):
        return {"sample": index}


def test_dense_manifest_can_reuse_nearest_sparse_vlm_cache(tmp_path):
    cache = tmp_path / "cache" / "Scenario" / "Route"
    cache.mkdir(parents=True)
    np.save(cache / "0000.npy", np.asarray([0.0], dtype=np.float16))
    np.save(cache / "0002.npy", np.asarray([2.0], dtype=np.float16))
    full_manifest = tmp_path / "full.jsonl"
    sparse_manifest = tmp_path / "sparse.jsonl"
    entries = [
        {"scenario": "Scenario", "route": "Route", "frame": f"{frame:04d}"}
        for frame in range(3)
    ]
    full_manifest.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    sparse_manifest.write_text(
        "".join(json.dumps(entries[index]) + "\n" for index in (0, 2))
    )
    dataset = VLMIntentDataset(
        _FakeCarlaDataset(),
        vlm_cache_dir=str(tmp_path / "cache"),
        manifest_path=str(full_manifest),
        nearest_cache_manifest_path=str(sparse_manifest),
    )
    assert len(dataset) == 3
    # The middle frame is equidistant and deterministically reuses the earlier one.
    assert dataset[1]["vlm_hidden"].item() == 0.0
