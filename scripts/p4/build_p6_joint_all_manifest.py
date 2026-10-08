"""Combine the route-disjoint P6 manifests for final all-frame training."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def build_all_manifest(split_dir: Path, output: Path) -> int:
    metadata = json.loads((split_dir / "metadata.json").read_text())
    sources = (split_dir / "train.jsonl", split_dir / "heldout.jsonl")
    expected = (metadata["train_frames"], metadata["heldout_frames"])
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts = []
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as destination:
            temporary = Path(destination.name)
            for source in sources:
                count = 0
                with source.open(encoding="utf-8") as stream:
                    for line in stream:
                        entry = json.loads(line)
                        key = entry["key"]
                        if key in seen:
                            raise ValueError(f"Duplicate frame key in P6 split: {key}")
                        seen.add(key)
                        destination.write(line if line.endswith("\n") else line + "\n")
                        count += 1
                counts.append(count)
            if tuple(counts) != expected or len(seen) != metadata["total_frames"]:
                raise ValueError(
                    f"P6 split counts {counts} do not match metadata "
                    f"{expected} / {metadata['total_frames']}"
                )
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    count = build_all_manifest(args.split, args.out)
    print(f"P6 all-frame manifest: {count} unique frames -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
