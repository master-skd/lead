"""Opt-in per-process stage timing; never introduces distributed collectives."""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


class StageTimingRecorder:
    def __init__(
        self,
        directory,
        component,
        rank,
        device=None,
        *,
        enabled=False,
        every=20,
        synchronize=True,
    ):
        if every < 1:
            raise ValueError("timing interval must be positive")
        self.enabled = enabled
        self.every = every
        self.synchronize = synchronize
        self.device = device
        self.component = component
        self.rank = int(rank)
        self.run_id = os.environ.get("LEAD_TIMING_RUN_ID", f"pid-{os.getpid()}")
        self.path = Path(directory) / f"{component}_rank{rank}.jsonl"
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_environment(cls, component, rank, device=None):
        return cls(
            os.environ.get("LEAD_TIMING_DIR", "outputs/timing"),
            component,
            rank,
            device,
            enabled=os.environ.get("LEAD_TRAIN_TIMING", "0") == "1",
            every=int(os.environ.get("LEAD_TIMING_EVERY", "20")),
            synchronize=os.environ.get("LEAD_TIMING_SYNC", "1") == "1",
        )

    def start(self, event, index=None, *, force=False, **metadata):
        active = self.enabled and (
            force or index is None or index < 5 or index % self.every == 0
        )
        return StageTiming(self, active, event, metadata)

    def sync(self):
        if self.synchronize and self.device is not None:
            import torch

            torch.cuda.synchronize(self.device)


class StageTiming:
    def __init__(self, recorder, active, event, metadata):
        self.recorder = recorder
        self.active = active
        self.stages = {}
        self.offsets = {}
        self.metadata = metadata
        self.event = event
        self.started = time.perf_counter() if active else 0.0
        self.timestamp = time.time() if active else 0.0

    @contextmanager
    def phase(self, name, *, gpu=False):
        if not self.active:
            yield
            return
        # Attribute pending work from the previous phase to that phase, not this one.
        if gpu:
            self.recorder.sync()
        begin = time.perf_counter()
        self.offsets[name] = (begin - self.started) * 1000
        try:
            yield
        finally:
            if gpu:
                self.recorder.sync()
            self.stages[name] = (time.perf_counter() - begin) * 1000

    def finish(self, **metadata):
        if not self.active:
            return
        record = dict(
            event=self.event,
            component=self.recorder.component,
            rank=self.recorder.rank,
            pid=os.getpid(),
            run_id=self.recorder.run_id,
            timestamp=self.timestamp,
            cuda_synchronized=self.recorder.synchronize,
            total_ms=(time.perf_counter() - self.started) * 1000,
            stages_ms=self.stages,
            stage_start_ms=self.offsets,
            **self.metadata,
        )
        record.update(metadata)
        # Open/close on each sampled record: logs survive interrupted jobs and
        # persistent dataloader workers cannot inherit a live output descriptor.
        with self.recorder.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
