"""Summarize B3b shadow decisions and visualize a triggered planning tick."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import zlib
from pathlib import Path

import numpy as np


def decode_intent_probability_map(record: dict) -> np.ndarray:
    encoded = record["intent_probability_map"]
    if encoded["encoding"] != "uint8_zlib_base64":
        raise ValueError(f"Unsupported intent encoding: {encoded['encoding']}")
    shape = tuple(int(dim) for dim in encoded["shape"])
    if len(shape) != 2 or any(dim <= 0 for dim in shape):
        raise ValueError(f"Invalid intent shape: {shape}")
    expected_size = int(np.prod(shape))
    decompressor = zlib.decompressobj()
    raw = decompressor.decompress(base64.b64decode(encoded["data"]), expected_size + 1)
    if len(raw) != expected_size or not decompressor.eof:
        raise ValueError("Intent map payload does not match its shape")
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).astype(np.float32) / 255.0


def load_rows(route_dir: Path) -> list[dict]:
    path = route_dir / "local_path_shadow.jsonl"
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def summarize(route_dir: Path, rows: list[dict]) -> None:
    expanded = [row for row in rows if row["candidate_evaluated"]]
    safe = [
        (row, candidate)
        for row in expanded
        for candidate in row["candidates"][1:]
        if not candidate["predicted_collision"]
    ]
    corridor_ok = [
        (row, candidate) for row, candidate in safe
        if candidate["corridor_mean"] <= 0.1 and candidate["corridor_max"] <= 0.25
    ]
    pid_ok = [
        (row, candidate) for row, candidate in safe
        if candidate["pid_steer_effective"]
        and abs(candidate["steer_delta_from_baseline"]) <= 0.25
    ]
    print(
        f"{route_dir.name}: ticks={len(rows)} raw-risk={sum(row['raw_predicted_collision'] for row in rows)} "
        f"expanded={len(expanded)} predicted-safe={len(safe)} "
        f"corridor-pass={len(corridor_ok)} PID-pass={len(pid_ok)} "
        f"would-switch={sum(row['would_switch'] for row in rows)}"
    )
    infraction_path = route_dir / "infractions.json"
    if infraction_path.exists():
        with infraction_path.open(encoding="utf-8") as file:
            infractions = json.load(file)["infractions"]
        for infraction in infractions:
            if "COLLISION" not in infraction.get("event_type", ""):
                continue
            step = int(infraction["step"])
            preceding = [
                row for row in rows if step - 40 <= row["step"] < step
                and row["raw_predicted_collision"]
            ]
            print(
                f"  actual collision step={step}; risk in prior 40 ticks="
                f"{[row['step'] for row in preceding]}"
            )


def plot_tick(row: dict, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from lead.training.config_training import TrainingConfig

    if not row["candidate_evaluated"] or "intent_probability_map" not in row:
        raise ValueError("Plotting needs a schema-v2 expanded shadow row")
    config = TrainingConfig()
    probability = decode_intent_probability_map(row)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(
        probability, origin="lower", cmap="viridis", vmin=0, vmax=1,
        extent=(config.min_x_meter, config.max_x_meter,
                config.min_y_meter, config.max_y_meter),
    )
    colors = ("white", "red", "orange", "cyan", "magenta")
    for candidate in row["candidates"]:
        index = candidate["index"]
        route = np.asarray(candidate["route_xy_m"])
        ego = np.asarray(candidate["predicted_ego_xy_m"])
        ax.plot(route[:, 0], route[:, 1], color=colors[index], alpha=0.75,
                label=f"{candidate['offset_m']:+.2f}m c={candidate['corridor_mean']:.2f} "
                      f"hit={candidate['predicted_collision']}")
        ax.scatter(ego[:, 0], ego[:, 1], color=colors[index], s=9)
    for actor in row["predicted_actors"]:
        xy = np.asarray(actor["xy_m"])
        ax.plot(xy[:, 0], xy[:, 1], "k--", alpha=0.6)
        ax.scatter(xy[0, 0], xy[0, 1], color="black", s=20)
    ax.scatter(0, 0, marker="x", color="white", s=100)
    ax.set(xlabel="ego-forward x (m)", ylabel="ego-lateral y (m)",
           title=f"step {row['step']}: intent probability, paths, predicted actors")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="local_evaluation_* directory or one route directory")
    parser.add_argument("--route", help="route ID for --plot-step")
    parser.add_argument("--plot-step", type=int)
    parser.add_argument("--plot-output", type=Path)
    args = parser.parse_args()
    route_dirs = (
        [args.root] if (args.root / "local_path_shadow.jsonl").exists()
        else sorted(path for path in args.root.iterdir()
                    if (path / "local_path_shadow.jsonl").exists())
    )
    if not route_dirs:
        parser.error("No local_path_shadow.jsonl found")
    for route_dir in route_dirs:
        rows = load_rows(route_dir)
        summarize(route_dir, rows)
        if args.plot_step is not None and (args.route is None or args.route == route_dir.name):
            matched = [row for row in rows if row["step"] == args.plot_step]
            if not matched:
                parser.error(f"No step {args.plot_step} in route {route_dir.name}")
            output_path = args.plot_output or Path(f"b3b_{route_dir.name}_step{args.plot_step}.png")
            plot_tick(matched[0], output_path)
            print(f"  plot -> {output_path}")


if __name__ == "__main__":
    main()
