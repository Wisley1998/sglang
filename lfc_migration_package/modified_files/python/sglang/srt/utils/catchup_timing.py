"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
LFC (Linear Factor Caching) Catch-up Stage Timing Instrumentation.

This module provides timing infrastructure for measuring the time breakdown
of the SSM catch-up stage when prefix cache hits occur with misalignment (Δ > 0).
"""

import csv
import json
import logging
import os
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class CatchupTimingRecord:
    """A single timing record for catch-up stage measurement."""

    # Batch information
    batch_size: int = 0
    total_tokens: int = 0
    prefix_len: int = 0
    delta: int = 0  # misalignment from snapshot boundary
    snapshot_interval: int = 500  # K value

    # Timing measurements (in milliseconds)
    prefix_match_ms: float = 0.0
    ssm_state_load_ms: float = 0.0
    catchup_recompute_ms: float = 0.0
    total_prefill_ms: float = 0.0

    # Mode information
    mode: str = "baseline"  # "baseline" or "lfc"

    # Additional metadata
    layer_id: int = -1
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class CUDATimer:
    """CUDA event-based timer for accurate GPU timing."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda")
        self._enabled = torch.cuda.is_available()
        if self._enabled:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        """Record start time."""
        if self._enabled:
            self.start_event.record()

    def stop(self) -> float:
        """Record end time and return elapsed time in milliseconds."""
        if not self._enabled:
            return 0.0
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)


class CatchupTimingCollector:
    """Collects and manages timing records for LFC catch-up stage measurements."""

    _instance: Optional["CatchupTimingCollector"] = None
    _lock = threading.Lock()

    def __init__(self, output_dir: str = "/tmp/sglang_catchup"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.records: List[CatchupTimingRecord] = []
        self._current_record: Optional[CatchupTimingRecord] = None
        self._phase_timers: Dict[str, CUDATimer] = {}
        self._phase_start_times: Dict[str, float] = {}
        self._enabled = True
        self._lock = threading.Lock()

        # Initialize CUDA timers for each phase
        self._init_timers()

        logger.info(f"CatchupTimingCollector initialized. Output dir: {self.output_dir}")

    @classmethod
    def get_instance(cls, output_dir: str = "/tmp/sglang_catchup") -> "CatchupTimingCollector":
        """Get or create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(output_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance."""
        with cls._lock:
            cls._instance = None

    def _init_timers(self):
        """Initialize CUDA timers for all timing phases."""
        phases = ["prefix_match", "ssm_state_load", "catchup_recompute", "total_prefill"]
        for phase in phases:
            self._phase_timers[phase] = CUDATimer()

    def enable(self):
        """Enable timing collection."""
        self._enabled = True

    def disable(self):
        """Disable timing collection."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if timing collection is enabled."""
        return self._enabled

    def start_record(
        self,
        batch_size: int = 0,
        total_tokens: int = 0,
        prefix_len: int = 0,
        delta: int = 0,
        snapshot_interval: int = 500,
        mode: str = "baseline",
        layer_id: int = -1,
    ):
        """Start a new timing record."""
        if not self._enabled:
            return

        with self._lock:
            self._current_record = CatchupTimingRecord(
                batch_size=batch_size,
                total_tokens=total_tokens,
                prefix_len=prefix_len,
                delta=delta,
                snapshot_interval=snapshot_interval,
                mode=mode,
                layer_id=layer_id,
            )
            self._phase_start_times.clear()

    def end_record(self):
        """End the current timing record and save it."""
        if not self._enabled or self._current_record is None:
            return

        with self._lock:
            self.records.append(self._current_record)
            self._current_record = None

    @contextmanager
    def time_phase(self, phase_name: str):
        """Context manager for timing a specific phase.

        Usage:
            with collector.time_phase("prefix_match"):
                # code to time
        """
        if not self._enabled or self._current_record is None:
            yield
            return

        timer = self._phase_timers.get(phase_name)
        if timer is None:
            timer = CUDATimer()
            self._phase_timers[phase_name] = timer

        timer.start()
        try:
            yield
        finally:
            elapsed_ms = timer.stop()
            self._set_phase_time(phase_name, elapsed_ms)

    def start_phase(self, phase_name: str):
        """Start timing a phase (alternative to context manager)."""
        if not self._enabled or self._current_record is None:
            return

        timer = self._phase_timers.get(phase_name)
        if timer is None:
            timer = CUDATimer()
            self._phase_timers[phase_name] = timer
        timer.start()

    def end_phase(self, phase_name: str):
        """End timing a phase (alternative to context manager)."""
        if not self._enabled or self._current_record is None:
            return

        timer = self._phase_timers.get(phase_name)
        if timer is not None:
            elapsed_ms = timer.stop()
            self._set_phase_time(phase_name, elapsed_ms)

    def _set_phase_time(self, phase_name: str, elapsed_ms: float):
        """Set the time for a specific phase in the current record."""
        if self._current_record is None:
            return

        phase_attr = f"{phase_name}_ms"
        if hasattr(self._current_record, phase_attr):
            setattr(self._current_record, phase_attr, elapsed_ms)

    def add_complete_record(self, record: CatchupTimingRecord):
        """Add a complete record directly."""
        if not self._enabled:
            return
        with self._lock:
            self.records.append(record)

    def clear_records(self):
        """Clear all collected records."""
        with self._lock:
            self.records.clear()

    def get_records(self) -> List[CatchupTimingRecord]:
        """Get a copy of all records."""
        with self._lock:
            return list(self.records)

    def compute_statistics(self) -> Dict:
        """Compute statistics (mean, p50, p99) for all timing phases."""
        if not self.records:
            return {}

        stats = {}
        phases = ["prefix_match_ms", "ssm_state_load_ms", "catchup_recompute_ms", "total_prefill_ms"]

        for mode in ["baseline", "lfc"]:
            mode_records = [r for r in self.records if r.mode == mode]
            if not mode_records:
                continue

            stats[mode] = {}
            for phase in phases:
                values = [getattr(r, phase) for r in mode_records]
                if values:
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    stats[mode][phase] = {
                        "mean": statistics.mean(values),
                        "p50": sorted_values[int(n * 0.5)],
                        "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1],
                        "min": min(values),
                        "max": max(values),
                        "count": n,
                    }

        return stats

    def compute_speedup(self) -> Dict:
        """Compute speedup of LFC vs baseline."""
        stats = self.compute_statistics()

        if "baseline" not in stats or "lfc" not in stats:
            return {}

        speedup = {}
        for phase in stats.get("baseline", {}):
            baseline_mean = stats["baseline"][phase]["mean"]
            lfc_mean = stats["lfc"][phase]["mean"]
            if lfc_mean > 0:
                speedup[phase] = {
                    "speedup": baseline_mean / lfc_mean,
                    "baseline_ms": baseline_mean,
                    "lfc_ms": lfc_mean,
                }

        return speedup

    def export_csv(self, filename: str = "catchup_results.csv") -> str:
        """Export records to CSV file."""
        filepath = self.output_dir / filename

        with open(filepath, "w", newline="") as f:
            if self.records:
                fieldnames = list(self.records[0].to_dict().keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for record in self.records:
                    writer.writerow(record.to_dict())

        logger.info(f"Exported {len(self.records)} records to {filepath}")
        return str(filepath)

    def export_json(self, filename: str = "catchup_results.json") -> str:
        """Export records and statistics to JSON file."""
        filepath = self.output_dir / filename

        data = {
            "records": [r.to_dict() for r in self.records],
            "statistics": self.compute_statistics(),
            "speedup": self.compute_speedup(),
            "metadata": {
                "total_records": len(self.records),
                "export_time": time.time(),
            }
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported results to {filepath}")
        return str(filepath)

    def print_summary(self):
        """Print a summary of collected timing data."""
        if not self.records:
            print("No timing records collected.")
            return

        stats = self.compute_statistics()
        speedup = self.compute_speedup()

        print("\n" + "=" * 60)
        print("LFC Catch-up Stage Timing Summary")
        print("=" * 60)
        print(f"Total records: {len(self.records)}")

        for mode, mode_stats in stats.items():
            print(f"\n{mode.upper()} Mode:")
            for phase, phase_stats in mode_stats.items():
                print(f"  {phase}:")
                print(f"    Mean: {phase_stats['mean']:.3f} ms")
                print(f"    P50:  {phase_stats['p50']:.3f} ms")
                print(f"    P99:  {phase_stats['p99']:.3f} ms")

        if speedup:
            print("\nSpeedup (Baseline / LFC):")
            for phase, data in speedup.items():
                print(f"  {phase}: {data['speedup']:.2f}x")

        print("=" * 60 + "\n")


# Global instance accessor
def get_timing_collector() -> Optional[CatchupTimingCollector]:
    """Get the global timing collector if timing is enabled."""
    from sglang.srt.environ import envs

    if envs.SGLANG_CATCHUP_TIMING.value:
        output_dir = envs.SGLANG_CATCHUP_TIMING_DIR.value
        return CatchupTimingCollector.get_instance(output_dir)
    return None


def is_timing_enabled() -> bool:
    """Check if timing collection is enabled via environment variable."""
    from sglang.srt.environ import envs
    return envs.SGLANG_CATCHUP_TIMING.value


def is_lfc_enabled() -> bool:
    """Check if LFC mode is enabled via environment variable."""
    from sglang.srt.environ import envs
    return envs.SGLANG_LFC_ENABLED.value


def get_snapshot_interval() -> int:
    """Get the snapshot interval (K value) from environment variable."""
    from sglang.srt.environ import envs
    return envs.SGLANG_SNAPSHOT_INTERVAL.value
