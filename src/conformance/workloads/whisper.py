"""Workload 2 — batch speech transcription.

Where LoRA stresses GPU memory and long-running compute, this stresses the axis
LoRA barely touches: getting bulk data *into* a sandbox and artifacts *out*. A GPU
platform that can train but cannot move a dataset is unusable for real work, and
this is the failure that tends to surface only at customer-onboarding time.

The Master Spec's section 29 is directly relevant: the only upload path used to be
a ~10MB `.ipynb` endpoint, and the base64-in-a-shell-command trick in
`workload_script_builder.py` is silently capped by the target shell's ARG_MAX well
below dataset size. SFTP upload was added to fix that. This workload is how you
find out whether the fix holds for a real audio corpus.

Data: LibriSpeech ASR (`openslr/librispeech_asr`, clean split) — real read speech
with reference transcripts, so word error rate is computable rather than
eyeballed. A transcription job that runs but produces garbage passes a throughput
benchmark and fails this.
"""

from __future__ import annotations

import json
import re
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

app = typer.Typer(add_completion=False, help="Batch Whisper transcription with WER scoring")

_NORMALISE = re.compile(r"[^a-z0-9' ]+")

CANARY_PAIRS = [
    ("the quick brown fox jumps over the lazy dog", "the quick brown fox jumps over the lazy dog"),
    ("hello world", "hello  World!"),
    ("she sells sea shells", "she sells sea shell"),
]


def normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation, collapse whitespace. WER is meaningless without it."""
    return _NORMALISE.sub(" ", text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words, divided by reference length.

    Iterative two-row DP rather than the textbook full matrix: a long transcript
    against a long hypothesis is O(n*m) in memory otherwise, and this workload is
    supposed to stress the platform, not the harness.
    """
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(
                previous[j - 1] if r == h else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        previous = current
    return previous[-1] / len(ref)


def corpus_wer(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """Aggregate WER plus the distribution, which is where problems hide."""
    if not pairs:
        return {"n": 0, "wer": 0.0}
    rates = [word_error_rate(ref, hyp) for ref, hyp in pairs]
    empties = sum(1 for _, hyp in pairs if not normalise(hyp))
    ordered = sorted(rates)
    return {
        "n": len(pairs),
        "wer": round(sum(rates) / len(rates), 4),
        "wer_median": round(ordered[len(ordered) // 2], 4),
        "wer_p90": round(ordered[int(0.9 * (len(ordered) - 1))], 4),
        "empty_transcripts": empties,
        # A run whose mean WER looks fine but has a fat tail of empty outputs is
        # a partially-broken run, and the mean alone will not show it.
        "empty_rate": round(empties / len(pairs), 4),
    }


def _run_canary(output_dir: Path, started: float) -> int:
    device, vram = detect_device()
    timings = Timings()
    with timings.stage("canary"):
        metrics = corpus_wer(CANARY_PAIRS)
        assert word_error_rate("a b c", "a b c") == 0.0
        assert word_error_rate("a b c", "") == 1.0
    WorkloadManifest(
        workload="whisper_batch_transcribe",
        started_at_unix=started,
        duration_seconds=time.time() - started,
        canary=True,
        device=device,
        config={"vram_gib": round(vram, 2), "pairs": len(CANARY_PAIRS)},
        metrics=metrics,
        timings_seconds=timings.to_dict(),
    ).write(output_dir)
    return 0


def select_model(vram_gib: float, override: str | None) -> str:
    if override:
        return override
    if vram_gib >= 24:
        return "openai/whisper-large-v3"
    if vram_gib >= 10:
        return "openai/whisper-medium"
    if vram_gib > 0:
        return "openai/whisper-small"
    return "openai/whisper-tiny"


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
    output_dir: str = typer.Option("artifacts", help="Where to write manifest and transcripts"),
    clips: int = typer.Option(200, help="Number of audio clips to transcribe"),
    model_id: str = typer.Option(None, help="Override the auto-selected Whisper model"),
    dataset_id: str = typer.Option("openslr/librispeech_asr", help="HF dataset id"),
    dataset_config: str = typer.Option("clean", help="Dataset config name"),
    split: str = typer.Option("validation", help="Dataset split"),
    batch_size: int = typer.Option(8, help="Clips per forward pass"),
    local_audio_dir: str = typer.Option(
        None, help="Transcribe local files instead of downloading (tests SFTP upload path)"
    ),
    canary: bool = typer.Option(False, help="Sub-minute route check; no download, no GPU"),
) -> None:
    """Transcribe a batch of audio and score word error rate against references."""
    started = time.time()
    out = resolve_output_dir(output_dir)
    if canary:
        raise typer.Exit(_run_canary(out, started))

    import torch
    from transformers import pipeline

    device, vram = detect_device()
    resolved_model = select_model(vram, model_id)
    timings = Timings()

    with timings.stage("data"):
        if local_audio_dir:
            # The interesting path: audio that arrived via SFTP upload rather than
            # egress. Proves data actually landed on the sandbox intact.
            audio_dir = Path(local_audio_dir)
            files = sorted(
                p for p in audio_dir.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".mp3"}
            )[:clips]
            if not files:
                raise typer.BadParameter(f"no audio files under {audio_dir}")
            samples: list[dict[str, Any]] = [{"path": str(p), "reference": ""} for p in files]
        else:
            from datasets import load_dataset

            raw = load_dataset(dataset_id, dataset_config, split=split, streaming=False)
            raw = raw.select(range(min(clips, len(raw))))
            samples = [{"audio": row["audio"], "reference": row.get("text", "")} for row in raw]

    print(f"[whisper] {len(samples)} clips, model={resolved_model}, device={device}", flush=True)

    with timings.stage("load_model"):
        asr = pipeline(
            "automatic-speech-recognition",
            model=resolved_model,
            device=0 if device == "cuda" else -1,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

    transcripts: list[dict[str, Any]] = []
    with timings.stage("transcribe"):
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            inputs = [c["path"] if "path" in c else c["audio"] for c in chunk]
            outputs = asr(inputs, batch_size=len(chunk))
            if isinstance(outputs, dict):
                outputs = [outputs]
            for sample, result in zip(chunk, outputs, strict=True):
                transcripts.append(
                    {
                        "reference": sample["reference"],
                        "hypothesis": (result or {}).get("text", ""),
                    }
                )
            emit_progress(len(transcripts), len(samples))

    scored = [(t["reference"], t["hypothesis"]) for t in transcripts if t["reference"]]
    metrics = corpus_wer(scored) if scored else {"n": 0, "note": "no references available"}

    transcript_path = out / "transcripts.jsonl"
    with transcript_path.open("w") as fh:
        for row in transcripts:
            fh.write(json.dumps(row) + "\n")

    elapsed = timings.to_dict().get("transcribe", 0.0) or 1.0
    metrics["clips_per_second"] = round(len(transcripts) / elapsed, 3)

    WorkloadManifest(
        workload="whisper_batch_transcribe",
        started_at_unix=started,
        duration_seconds=time.time() - started,
        device=device,
        config={
            "model_id": resolved_model,
            "clips": len(samples),
            "batch_size": batch_size,
            "source": local_audio_dir or f"{dataset_id}/{dataset_config}/{split}",
            "vram_gib": round(vram, 2),
        },
        metrics=metrics,
        timings_seconds=timings.to_dict(),
        artifacts=[str(transcript_path)],
        completed_steps=len(transcripts),
        total_steps=len(samples),
        peak_vram_gib=peak_vram_gib(),
    ).write(out)
    print(f"[whisper] {json.dumps(metrics)}", flush=True)


if __name__ == "__main__":
    app()
