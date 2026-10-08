import json

import pytest

from scripts.p4.build_p6_joint_all_manifest import build_all_manifest


def _write_split(split, train_keys, heldout_keys):
    split.mkdir()
    for name, keys in (("train", train_keys), ("heldout", heldout_keys)):
        (split / f"{name}.jsonl").write_text(
            "".join(json.dumps({"key": key}) + "\n" for key in keys)
        )
    (split / "metadata.json").write_text(json.dumps({
        "train_frames": len(train_keys),
        "heldout_frames": len(heldout_keys),
        "total_frames": len(train_keys) + len(heldout_keys),
    }))


def test_all_manifest_combines_disjoint_frames(tmp_path):
    split = tmp_path / "split"
    output = tmp_path / "all" / "all_frames.jsonl"
    _write_split(split, ["a", "b"], ["c"])

    assert build_all_manifest(split, output) == 3
    assert [json.loads(line)["key"] for line in output.read_text().splitlines()] == [
        "a", "b", "c",
    ]


def test_all_manifest_rejects_overlap_without_replacing_output(tmp_path):
    split = tmp_path / "split"
    output = tmp_path / "all_frames.jsonl"
    output.write_text("existing\n")
    _write_split(split, ["a"], ["a"])

    with pytest.raises(ValueError, match="Duplicate frame key"):
        build_all_manifest(split, output)
    assert output.read_text() == "existing\n"
