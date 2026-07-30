from pathlib import Path

import pytest

from conformance.workloads.lora_config import RunConfig, resolve_config, select_tier


@pytest.mark.parametrize(
    ("vram", "expected"),
    [
        (80.0, "large"),
        (40.0, "large"),
        (39.9, "medium"),
        (24.0, "medium"),
        (18.0, "medium"),
        (17.9, "small"),
        (10.0, "small"),
        (9.9, "tiny"),
        (0.0, "tiny"),
    ],
)
def test_tier_boundaries(vram: float, expected: str) -> None:
    assert select_tier(vram)["tier"] == expected


def test_tiny_tier_quantizes_but_large_does_not() -> None:
    assert select_tier(0.0)["load_in_4bit"] is True
    assert select_tier(80.0)["load_in_4bit"] is False


def test_effective_batch_is_stable_enough_across_tiers() -> None:
    # Effective batch should not swing wildly between tiers, or learning-rate
    # behaviour stops being comparable across sandbox sizes.
    batches = {
        select_tier(v)["batch_size"] * select_tier(v)["grad_accum"] for v in (0.0, 12.0, 24.0, 80.0)
    }
    assert max(batches) / min(batches) <= 2


def test_resolve_config_applies_overrides_over_tier_defaults() -> None:
    cfg = resolve_config(vram_gib=80.0, overrides={"train_rows": 50, "model_id": "tiny/model"})
    assert cfg.tier == "large"
    assert cfg.train_rows == 50
    assert cfg.model_id == "tiny/model"


def test_resolve_config_ignores_none_overrides() -> None:
    cfg = resolve_config(vram_gib=24.0, overrides={"train_rows": None})
    assert cfg.train_rows == RunConfig.__dataclass_fields__["train_rows"].default


def test_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKLOAD_TRAIN_ROWS", "123")
    assert resolve_config(vram_gib=24.0).train_rows == 123


def test_explicit_override_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKLOAD_TRAIN_ROWS", "123")
    assert resolve_config(vram_gib=24.0, overrides={"train_rows": 7}).train_rows == 7


def test_cpu_device_when_no_vram() -> None:
    cfg = resolve_config(vram_gib=0.0)
    assert cfg.device == "cpu"
    assert cfg.tier == "tiny"


def test_output_dir_coerced_to_path() -> None:
    cfg = resolve_config(vram_gib=24.0, overrides={"output_dir": "somewhere"})
    assert isinstance(cfg.output_dir, Path)
    assert cfg.to_dict()["output_dir"] == "somewhere"
