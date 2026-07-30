"""Run manifest and metrics output.

A workload that leaves no evidence behind is useless as a platform test. Every
run writes a manifest describing what hardware it got, what it decided to do,
how long each stage took, and what it produced.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .lora_config import RunConfig


class Timings:
    """Accumulates wall-clock durations per named stage."""

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
    return out.stdout.strip() or None if out.returncode == 0 else None


def peak_vram_gib() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1024**3, 2)


def write_manifest(
    cfg: RunConfig,
    *,
    baseline: dict[str, Any] | None,
    tuned: dict[str, Any] | None,
    timings: dict[str, float],
    started_at: float,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write artifacts/manifest.json and return its path."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "workload": "banking77-intent-json-lora",
        "git_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "started_at_unix": round(started_at, 3),
        "duration_seconds": round(time.time() - started_at, 2),
        "config": cfg.to_dict(),
        "timings_seconds": timings,
        "peak_vram_gib": peak_vram_gib(),
        "metrics": {"baseline": baseline, "tuned": tuned},
    }
    if baseline and tuned:
        manifest["delta"] = {
            "json_valid_rate": round(tuned["json_valid_rate"] - baseline["json_valid_rate"], 4),
            "accuracy": round(tuned["accuracy"] - baseline["accuracy"], 4),
        }
    if extra:
        manifest.update(extra)

    path = cfg.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def write_predictions(
    cfg: RunConfig,
    name: str,
    records: list[dict[str, Any]],
    predictions: list[str],
) -> Path:
    """Dump raw model outputs alongside gold labels for eyeball inspection."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.output_dir / f"predictions_{name}.jsonl"
    with path.open("w") as fh:
        for record, raw in zip(records, predictions, strict=True):
            fh.write(
                json.dumps({"text": record["text"], "gold": record["label"], "raw": raw}) + "\n"
            )
    return path
