"""Workload 1 — LoRA fine-tuning for structured intent extraction.

Real task, real data: banking77 is 13,083 genuine customer-support messages sent
to a retail bank, labelled across 77 intents. The model learns to emit strict JSON
naming the intent, which gives two metrics that move independently — accuracy, and
whether the output is machine-parseable at all.

Changes from the standalone version, all driven by the adversarial suite:
  - step-based checkpointing (was epoch-based, so a single-epoch run checkpointed
    exactly once, at the end, making kill-at-60% untestable);
  - `--resume-from`, with the resumed step recorded in the manifest so a silent
    restart-from-zero cannot pass as a successful resume;
  - `--emit-progress`, so the orchestrator knows when 60% is instead of guessing;
  - `--canary`, a sub-minute no-download mode for route conformance.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import typer

from .base import (
    Timings,
    WorkloadManifest,
    detect_device,
    emit_progress,
    peak_vram_gib,
    resolve_output_dir,
)
from .lora_config import resolve_config
from .lora_data import build_messages, build_training_record, load_splits, parse_prediction
from .lora_eval import generate_predictions, score

app = typer.Typer(add_completion=False, help="LoRA fine-tune: bank support message -> JSON intent")

CANARY_ROWS = [
    ("my card hasn't arrived yet", "card_arrival"),
    ("I lost my card yesterday", "lost_or_stolen_card"),
    ("the top up didn't go through", "top_up_failed"),
    ("when will my new card get here", "card_arrival"),
]
CANARY_LABELS = ["card_arrival", "lost_or_stolen_card", "top_up_failed"]


def _latest_checkpoint(training_dir: Path) -> tuple[Path | None, int]:
    """Find the highest-numbered HF checkpoint directory and its step."""
    if not training_dir.is_dir():
        return None, 0
    best: tuple[Path | None, int] = (None, 0)
    for child in training_dir.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            try:
                step = int(child.name.split("-", 1)[1])
            except ValueError:
                continue
            if step > best[1]:
                best = (child, step)
    return best


def _progress_callback(total_steps: int) -> Any:
    """A TrainerCallback that emits parseable progress and flushes each step."""
    from transformers import TrainerCallback

    class ProgressEmitter(TrainerCallback):  # type: ignore[misc]
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            emit_progress(int(state.global_step), total_steps, epoch=round(state.epoch or 0.0, 3))

    return ProgressEmitter()


def _sft_config(cfg: Any, output_dir: str, checkpoint_every: int, max_steps: int) -> Any:
    """Build SFTConfig, tolerating TRL's renamed fields across versions."""
    from trl import SFTConfig

    desired: dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "logging_steps": 5,
        # Step-based, not epoch-based. This is what makes kill-and-resume testable.
        "save_strategy": "steps",
        "save_steps": checkpoint_every,
        "save_total_limit": 3,
        "bf16": cfg.device == "cuda",
        "seed": cfg.seed,
        "report_to": [],
        "max_seq_length": cfg.max_seq_len,
        "max_length": cfg.max_seq_len,
        "packing": False,
        "gradient_checkpointing": True,
    }
    if max_steps > 0:
        desired["max_steps"] = max_steps
    else:
        desired["num_train_epochs"] = cfg.epochs

    known = {f.name for f in dataclasses.fields(SFTConfig)}
    dropped = sorted(set(desired) - known)
    if dropped:
        print(f"[lora] installed TRL does not accept {dropped}; skipping them.", flush=True)
    return SFTConfig(**{k: v for k, v in desired.items() if k in known})


def _run_canary(output_dir: Path, started: float) -> int:
    """Prove the route works without downloading a model or touching a GPU.

    Exercises the same import graph, prompt formatting, parser, and scorer the
    real run uses, so a canary pass means the route can carry the real workload.
    """
    device, vram = detect_device()
    timings = Timings()
    with timings.stage("canary"):
        records = [
            {"text": text, "label": label, "messages": build_messages(text, CANARY_LABELS)}
            for text, label in CANARY_ROWS
        ]
        train = [build_training_record(t, label, CANARY_LABELS) for t, label in CANARY_ROWS]
        # Simulate perfect predictions through the real parser and scorer.
        preds = [json.dumps({"intent": r["label"]}) for r in records]
        result = score(preds, [r["label"] for r in records], CANARY_LABELS)
        assert all(parse_prediction(p) for p in preds), "parser rejected its own format"
        assert len(train) == len(records)

    manifest = WorkloadManifest(
        workload="lora_intent_json",
        started_at_unix=started,
        duration_seconds=time.time() - started,
        canary=True,
        device=device,
        config={"vram_gib": round(vram, 2), "rows": len(records)},
        metrics=result.to_dict(),
        timings_seconds=timings.to_dict(),
    )
    manifest.write(output_dir)
    return 0


@app.callback()
def _main() -> None:
    """Keep typer in subcommand mode.

    With exactly one command and no callback, typer collapses the app and invokes
    that command directly — which makes `python -m ... run` fail with "unexpected
    extra argument (run)". Every route in this harness generates exactly that
    command line, so without this the whole matrix fails on the platform while
    passing every local unit test.
    """


@app.command()
def run(
    output_dir: str = typer.Option("artifacts", help="Where to write manifest and adapter"),
    train_rows: int = typer.Option(None, help="Cap on training examples"),
    eval_rows: int = typer.Option(None, help="Cap on evaluation examples"),
    model_id: str = typer.Option(None, help="Override the auto-selected model"),
    vram: float = typer.Option(None, help="Override detected VRAM in GiB"),
    checkpoint_every_steps: int = typer.Option(50, help="Save a checkpoint every N steps"),
    max_steps: int = typer.Option(0, help="Hard cap on training steps (0 = full epoch)"),
    resume_from: str = typer.Option(None, help="Training dir holding checkpoint-N to resume"),
    emit_progress_lines: bool = typer.Option(
        False, "--emit-progress", help="Print machine-readable progress each step"
    ),
    skip_baseline: bool = typer.Option(False, help="Skip the pre-tuning evaluation pass"),
    canary: bool = typer.Option(False, help="Sub-minute route check; no download, no GPU"),
) -> None:
    """Run the LoRA workload."""
    started = time.time()
    out = resolve_output_dir(output_dir)

    if canary:
        raise typer.Exit(_run_canary(out, started))

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM
    from trl import SFTTrainer

    from .lora_model import load_model, load_tokenizer, lora_config

    timings = Timings()
    cfg = resolve_config(
        vram_gib=vram,
        overrides={
            "train_rows": train_rows,
            "eval_rows": eval_rows,
            "model_id": model_id,
            "output_dir": str(out),
        },
    )
    print(f"[lora] config {json.dumps(cfg.to_dict())}", flush=True)

    with timings.stage("data"):
        train_records, eval_records, labels = load_splits(
            cfg.dataset_id, cfg.train_rows, cfg.eval_rows, cfg.seed
        )
    gold = [r["label"] for r in eval_records]
    tokenizer = load_tokenizer(cfg)

    baseline: dict[str, Any] | None = None
    if not skip_baseline:
        with timings.stage("baseline_eval"):
            base = load_model(cfg)
            preds = generate_predictions(base, tokenizer, eval_records, cfg)
            baseline = score(preds, gold, labels).to_dict()
            del base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"[lora] baseline {json.dumps(baseline)}", flush=True)

    training_dir = out / "training"
    resume_dir = Path(resume_from) if resume_from else None
    resume_ckpt, resumed_step = _latest_checkpoint(resume_dir) if resume_dir else (None, 0)
    if resume_dir and resume_ckpt is None:
        # Loud, not silent. A resume that quietly restarts from zero is the exact
        # false pass this scenario exists to catch.
        print(
            f"[lora] RESUME-FAILED: no checkpoint-* under {resume_dir}; starting from step 0",
            flush=True,
        )
    elif resume_ckpt:
        print(f"[lora] resuming from {resume_ckpt} at step {resumed_step}", flush=True)

    model_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16 if cfg.device == "cuda" else torch.float32,
        "device_map": "auto" if cfg.device == "cuda" else None,
    }
    if cfg.load_in_4bit and cfg.device == "cuda":
        from .lora_model import _quant_config

        model_kwargs["quantization_config"] = _quant_config()
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)
    except TypeError:
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)
    model.config.use_cache = False

    steps_per_epoch = max(1, len(train_records) // max(1, cfg.effective_batch))
    total_steps = max_steps if max_steps > 0 else steps_per_epoch

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": _sft_config(cfg, str(training_dir), checkpoint_every_steps, max_steps),
        "train_dataset": Dataset.from_list(train_records),
        "peft_config": lora_config(cfg),
    }
    try:
        trainer = SFTTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = SFTTrainer(**trainer_kwargs, tokenizer=tokenizer)

    if emit_progress_lines:
        trainer.add_callback(_progress_callback(total_steps))

    with timings.stage("train"):
        trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)

    completed = int(trainer.state.global_step)
    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with timings.stage("tuned_eval"):
        tuned_model = load_model(cfg, adapter_path=str(adapter_dir))
        preds = generate_predictions(tuned_model, tokenizer, eval_records, cfg)
        tuned = score(preds, gold, labels).to_dict()

    manifest = WorkloadManifest(
        workload="lora_intent_json",
        started_at_unix=started,
        duration_seconds=time.time() - started,
        device=cfg.device,
        config=cfg.to_dict(),
        metrics={"baseline": baseline, "tuned": tuned},
        timings_seconds=timings.to_dict(),
        artifacts=[str(adapter_dir), str(training_dir)],
        resumed_from_step=resumed_step or None,
        completed_steps=completed,
        total_steps=total_steps,
        peak_vram_gib=peak_vram_gib(),
    )
    manifest.write(out)
    print(f"[lora] tuned {json.dumps(tuned)}", flush=True)


if __name__ == "__main__":
    app()
