"""Concatenate independently clustered residual vocabularies into a nested one."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_vocab(sources: list[Path], output: Path) -> dict:
    if len(sources) < 2:
        raise ValueError("provide at least two source vocabularies")
    if output.suffix != ".npy":
        raise ValueError("output must have .npy suffix")
    centers = []
    shape = None
    for source in sources:
        array = np.load(source, allow_pickle=False)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError(f"expected finite [K,T] vocabulary: {source}")
        if shape is None:
            shape = array.shape[1]
        if array.shape[1] != shape:
            raise ValueError(f"profile horizon mismatch: {source}")
        centers.append(array.astype(np.float32))
    # Preserve every base center exactly and in order. Separate K-means runs are
    # not nested, so replacing K16 by K32 can discard useful braking profiles.
    nested = np.concatenate(centers, axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, nested)
    report = {
        "output": str(output.resolve()),
        "shape": list(nested.shape),
        "sources": [
            {"path": str(path.resolve()), "shape": list(array.shape)}
            for path, array in zip(sources, centers, strict=True)
        ],
        "prefix_preserved": bool(np.array_equal(nested[: len(centers[0])], centers[0])),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("sources", nargs="+", type=Path)
    args = parser.parse_args()
    report = build_vocab(args.sources, args.out)
    print(f"wrote {report['output']}: {report['shape']}")


if __name__ == "__main__":
    main()
