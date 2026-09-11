"""Create deterministic, scenario-stratified route splits for B2d experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def split_routes(
    entries: list[dict], heldout_fraction: float, seed: int,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Split whole routes, keeping at least one training route per scenario."""

    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be between zero and one")
    by_scenario: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        by_scenario[entry["scenario"]].add(entry["route"])

    train, heldout = set(), set()
    for scenario, route_set in sorted(by_scenario.items()):
        routes = sorted(
            route_set,
            key=lambda route: hashlib.sha256(
                f"{seed}:{scenario}:{route}".encode(),
            ).digest(),
        )
        if len(routes) < 2:
            num_heldout = 0
        else:
            num_heldout = min(len(routes) - 1, max(1, round(len(routes) * heldout_fraction)))
        heldout.update((scenario, route) for route in routes[:num_heldout])
        train.update((scenario, route) for route in routes[num_heldout:])
    return train, heldout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/p4/manifest.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--heldout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    with open(args.manifest) as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    train_routes, heldout_routes = split_routes(entries, args.heldout_fraction, args.seed)
    train_entries = [e for e in entries if (e["scenario"], e["route"]) in train_routes]
    heldout_entries = [e for e in entries if (e["scenario"], e["route"]) in heldout_routes]
    if train_routes & heldout_routes:
        raise RuntimeError("route leakage between train and heldout split")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train_entries), ("heldout", heldout_entries)):
        with (output_dir / f"{name}.jsonl").open("w") as handle:
            for entry in subset:
                handle.write(json.dumps(entry) + "\n")

    scenario_summary = {}
    scenarios = sorted({e["scenario"] for e in entries})
    for scenario in scenarios:
        scenario_summary[scenario] = {
            "train_routes": sum(s == scenario for s, _ in train_routes),
            "heldout_routes": sum(s == scenario for s, _ in heldout_routes),
            "train_frames": sum(e["scenario"] == scenario for e in train_entries),
            "heldout_frames": sum(e["scenario"] == scenario for e in heldout_entries),
        }
    metadata = {
        "seed": args.seed,
        "heldout_fraction_requested": args.heldout_fraction,
        "total_routes": len(train_routes) + len(heldout_routes),
        "train_routes": len(train_routes),
        "heldout_routes": len(heldout_routes),
        "total_frames": len(entries),
        "train_frames": len(train_entries),
        "heldout_frames": len(heldout_entries),
        "route_overlap": 0,
        "scenario_summary": scenario_summary,
    }
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps({k: v for k, v in metadata.items() if k != "scenario_summary"}, indent=2))


if __name__ == "__main__":
    main()
