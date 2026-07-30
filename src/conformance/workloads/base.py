"""The contract every real workload implements.

Three different workloads (LoRA, Whisper, FAISS) run through five different
delivery routes. If each emitted its own result shape, nothing could be compared
and the route matrix would need per-workload special cases. So every workload:

  - accepts `--canary`, completing in under ~60 seconds with no model download,
    which is what the route-conformance sweep uses so proving a route works costs
    seconds of GPU time rather than an hour;
  - writes `manifest.json` in the same schema;
  - prints `CONFORMANCE-MANIFEST: <path>` on the last line of stdout, so a route
    that can only capture stdout (papermill notebooks, CLI streaming) can still
    locate results.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

MANIFEST_MARKER = "CONFORMANCE-MANIFEST:"
PROGRESS_MARKER = "CONFORMANCE-PROGRESS:"


def emit_progress(step: int, total: int, **extra: Any) -> None:
    """Print a machine-readable progress line.

    The kill-at-60% scenario needs to know when 60% is, from outside, with only
    stdout to go on. Guessing with a wall-clock fraction would fire at the wrong
    point on a slower GPU and make the test meaningless.
    """
    payload = {"step": step, "total": total, "pct": round(100 * step / total, 2) if total else 0.0}
    payload.update(extra)
    print(f"{PROGRESS_MARKER} {json.dumps(payload)}", flush=True)


@dataclass(slots=True)
class WorkloadManifest:
    """Uniform result record for any workload."""

    workload: str
    started_at_unix: float
    duration_seconds: float = 0.0
    canary: bool = False
    device: str = "cpu"
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    timings_seconds: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    resumed_from_step: int | None = None
    completed_steps: int | None = None
    total_steps: int | None = None
    peak_vram_gib: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "canary": self.canary,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "git_sha": _git_sha(),
            "started_at_unix": round(self.started_at_unix, 3),
            "duration_seconds": round(self.duration_seconds, 2),
            "device": self.device,
            "config": self.config,
            "metrics": self.metrics,
            "timings_seconds": self.timings_seconds,
            "artifacts": self.artifacts,
            "resumed_from_step": self.resumed_from_step,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "peak_vram_gib": self.peak_vram_gib,
            "error": self.error,
        }

    def write(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "manifest.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        # Last line, deliberately: a route that can only scrape stdout tails it.
        print(f"{MANIFEST_MARKER} {path}", flush=True)
        return path


class Timings:
    def __init__(self) -> None:
        self._stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._stages[name] = round(time.perf_counter() - start, 2)

    def to_dict(self) -> dict[str, float]:
        return dict(self._stages)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def detect_device() -> tuple[str, float]:
    """Return (device, vram_gib). Never raises — a missing torch is a finding."""
    try:
        import torch
    except ImportError:
        return "cpu", 0.0
    if not torch.cuda.is_available():
        return "cpu", 0.0
    best = max(
        (
            torch.cuda.get_device_properties(i).total_memory
            for i in range(torch.cuda.device_count())
        ),
        default=0,
    )
    return "cuda", best / 1024**3


def peak_vram_gib() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1024**3, 2)


def resolve_output_dir(raw: str) -> Path:
    """Expand and create an output directory, honouring CONFORMANCE_OUTPUT_DIR."""
    path = Path(os.environ.get("CONFORMANCE_OUTPUT_DIR", raw)).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path
