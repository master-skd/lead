"""Summarize opt-in P6 timing logs without loading models or requiring CUDA."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def read_records(directory):
    records = []
    for path in sorted(directory.glob("*_rank*.jsonl")):
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Ignoring incomplete/invalid row: {path}:{number}")
    return records


def select_records(records, run_id=None, skip_first=10):
    if not records:
        return [], None
    if run_id is None:
        run_id = max(records, key=lambda row: row["timestamp"])["run_id"]
    selected = [row for row in records if row["run_id"] == run_id]
    train = [row for row in selected if row["event"] == "train_step"
             and row.get("status") != "loader_exhausted" and row["step"] >= skip_first]
    requests = {(row["rank"], row["request_index"]) for row in train}
    # Keep Qwen and training samples aligned; never mix validation or warm-up
    # requests into the steady-state service breakdown.
    qwen = [row for row in selected if row["event"] == "qwen_batch"
            and (row["rank"], row["request_index"]) in requests]
    lifecycle = [row for row in selected if row["event"] not in ("train_step", "qwen_batch")]
    return train + qwen + lifecycle, run_id


def print_stage_table(rows):
    values = defaultdict(list)
    for row in rows:
        for name, value in row["stages_ms"].items():
            values[(row["event"], name)].append(value / 1000)
        values[(row["event"], "TOTAL")].append(row["total_ms"] / 1000)
    print(f"{'event / stage':48s} {'n':>6s} {'mean(s)':>10s} {'p50(s)':>10s} {'p95(s)':>10s} {'max(s)':>10s}")
    for (event, stage), samples in sorted(values.items()):
        print(f"{event + ' / ' + stage:48s} {len(samples):6d} "
              f"{statistics.mean(samples):10.3f} {statistics.median(samples):10.3f} "
              f"{percentile(samples, .95):10.3f} {max(samples):10.3f}")


def print_rank_table(rows):
    ranks = defaultdict(list)
    for row in rows:
        if row["event"] == "train_step":
            ranks[row["rank"]].append(row)
    stages = ("data_wait", "qwen_request", "loss_and_labels", "backward_ddp")
    print("\nPer-rank means in seconds (backward includes DDP waiting):")
    print(f"{'rank':>4s} {'n':>6s} {'total':>10s} " + " ".join(f"{stage:>16s}" for stage in stages))
    for rank, samples in sorted(ranks.items()):
        total = statistics.mean(row["total_ms"] for row in samples) / 1000
        means = [statistics.mean(row["stages_ms"].get(stage, 0) for row in samples) / 1000
                 for stage in stages]
        print(f"{rank:4d} {len(samples):6d} {total:10.3f} " + " ".join(f"{value:16.3f}" for value in means))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="training logdir/timing")
    parser.add_argument("--skip-first", type=int, default=10, help="exclude first N steps of EACH epoch")
    parser.add_argument("--run-id", help="default: most recently recorded launch")
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error(f"timing directory does not exist: {args.directory}")
    rows, run_id = select_records(read_records(args.directory), args.run_id, args.skip_first)
    if not rows:
        parser.error("no records selected; check run ID or use --skip-first 0")
    print(f"run_id={run_id}; excluded first {args.skip_first} steps/epoch")
    print("Qwen stages are INSIDE qwen_request, not additional training time.")
    if all(row.get("cuda_synchronized", False) for row in rows):
        print("Sampled synchronized wall times include CPU work and device waits; no new DDP barriers are added.")
    else:
        print("WARNING: CUDA synchronization was disabled for some records; GPU stage times may reflect submission only.")
    print_stage_table(rows)
    print_rank_table(rows)


if __name__ == "__main__":
    main()
