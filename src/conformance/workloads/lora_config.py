"""VRAM-adaptive run configuration.

The sandbox platform exposes every GPU tier, so the workload must size itself at
runtime rather than assume a card. Everything in this module is deliberately
importable without torch installed, which keeps the tier logic unit-testable on
a laptop with no GPU.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Tier table: (minimum GiB of VRAM, config). Evaluated highest-first.
# Model IDs are pinned to a family with consistent chat templating so the same
# prompt-formatting code works across every tier.
_TIERS: tuple[tuple[float, dict[str, Any]], ...] = (
    (
        40.0,
        {
            "tier": "large",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "load_in_4bit": False,
            "batch_size": 16,
            "grad_accum": 1,
            "max_seq_len": 640,
            "lora_r": 32,
        },
    ),
    (
        18.0,
        {
            "tier": "medium",
            "model_id": "Qwen/Qwen2.5-3B-Instruct",
            "load_in_4bit": False,
            "batch_size": 8,
            "grad_accum": 2,
            "max_seq_len": 640,
            "lora_r": 16,
        },
    ),
    (
        10.0,
        {
            "tier": "small",
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "load_in_4bit": False,
            "batch_size": 4,
            "grad_accum": 4,
            "max_seq_len": 512,
            "lora_r": 16,
        },
    ),
    (
        0.0,
        {
            "tier": "tiny",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "load_in_4bit": True,
            "batch_size": 2,
            "grad_accum": 8,
            "max_seq_len": 512,
            "lora_r": 8,
        },
    ),
)


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Fully resolved settings for one workload run."""

    tier: str
    model_id: str
    load_in_4bit: bool
    batch_size: int
    grad_accum: int
    max_seq_len: int
    lora_r: int

    dataset_id: str = "PolyAI/banking77"
    train_rows: int = 4000
    eval_rows: int = 300
    epochs: float = 1.0
    learning_rate: float = 2e-4
    seed: int = 17
    output_dir: Path = field(default=Path("artifacts"))

    # Detected environment, recorded for the run manifest.
    detected_vram_gib: float = 0.0
    device: str = "cpu"

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["effective_batch"] = self.effective_batch
        return payload


def detect_vram_gib() -> tuple[float, str]:
    """Return (largest visible GPU VRAM in GiB, device string).

    Falls back to (0.0, "cpu") when torch is absent or no CUDA device is
    visible, which is itself a useful sandbox test result rather than a crash.
    """
    try:
        import torch
    except ImportError:
        return 0.0, "cpu"

    if not torch.cuda.is_available():
        return 0.0, "cpu"

    best = 0.0
    for idx in range(torch.cuda.device_count()):
        total_bytes = torch.cuda.get_device_properties(idx).total_memory
        best = max(best, total_bytes / (1024**3))
    return best, "cuda"


def select_tier(vram_gib: float) -> dict[str, Any]:
    """Pick the largest tier that fits in the given VRAM."""
    for threshold, cfg in _TIERS:
        if vram_gib >= threshold:
            return dict(cfg)
    # _TIERS ends at 0.0, so this is unreachable; kept as a guard against edits.
    raise RuntimeError(f"no tier matched vram_gib={vram_gib!r}")


def resolve_config(
    *,
    vram_gib: float | None = None,
    overrides: dict[str, Any] | None = None,
) -> RunConfig:
    """Build a RunConfig from detected hardware, then apply explicit overrides.

    Precedence, lowest to highest: tier defaults, WORKLOAD_* env vars, explicit
    overrides passed by the CLI.
    """
    device = "cpu"
    if vram_gib is None:
        vram_gib, device = detect_vram_gib()
    elif vram_gib > 0:
        device = "cuda"

    settings = select_tier(vram_gib)
    settings["detected_vram_gib"] = round(vram_gib, 2)
    settings["device"] = device

    for key in ("model_id", "dataset_id"):
        env_val = os.environ.get(f"WORKLOAD_{key.upper()}")
        if env_val:
            settings[key] = env_val
    for key in ("train_rows", "eval_rows", "batch_size", "max_seq_len"):
        env_val = os.environ.get(f"WORKLOAD_{key.upper()}")
        if env_val:
            settings[key] = int(env_val)

    if overrides:
        settings.update({k: v for k, v in overrides.items() if v is not None})

    if "output_dir" in settings:
        settings["output_dir"] = Path(settings["output_dir"])

    return RunConfig(**settings)
