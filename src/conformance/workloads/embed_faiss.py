"""Workload 3 — embedding generation and FAISS index build.

The RAG workload, and the one that stresses a different axis again: sustained
mixed CPU/GPU throughput, a large single artifact written to disk, and — uniquely
among the three — a correctness check that does not depend on a model being
*good*, only on the index being *right*.

That last property is why this workload earns its place. Recall@1 of a vector
index queried with the exact vectors it contains must be 1.0. Not "high" — exactly
1.0, as a matter of arithmetic. Any deviation means the index was corrupted:
truncated write, silent dtype downcast, a partially-flushed page cache on
teardown, non-determinism in the GPU reduction. Every one of those is a platform
fault rather than a model quality issue, and none of them shows up in a throughput
number.

Data: `sentence-transformers/natural-questions` — real search queries, so the
retrieval task is genuine rather than synthetic vectors.
"""

from __future__ import annotations

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

app = typer.Typer(add_completion=False, help="Embedding generation + FAISS index build")


def self_recall(index: Any, vectors: Any, k: int = 1) -> dict[str, Any]:
    """Query the index with the exact vectors it contains.

    Recall@1 must be exactly 1.0. Anything less is index corruption, not a
    quality signal — which makes this the cheapest possible integrity check on
    the platform's disk and GPU determinism.
    """
    scores, ids = index.search(vectors, k)
    expected = list(range(vectors.shape[0]))
    hits_at_1 = sum(1 for i, row in enumerate(ids) if row[0] == expected[i])
    hits_at_k = sum(1 for i, row in enumerate(ids) if expected[i] in row)
    return {
        "n_queries": int(vectors.shape[0]),
        "recall_at_1": round(hits_at_1 / vectors.shape[0], 6),
        "recall_at_k": round(hits_at_k / vectors.shape[0], 6),
        "k": k,
        "top1_score_mean": round(float(scores[:, 0].mean()), 6),
    }


def _run_canary(output_dir: Path, started: float) -> int:
    """Numpy-only integrity check: no model download, no GPU, no faiss required."""
    import numpy as np

    device, vram = detect_device()
    timings = Timings()
    with timings.stage("canary"):
        rng = np.random.default_rng(17)
        vectors = rng.normal(size=(64, 32)).astype("float32")
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        # Brute-force cosine search stands in for the FAISS flat index; the
        # invariant under test (self-recall == 1.0) is identical.
        sims = vectors @ vectors.T
        top1 = sims.argmax(axis=1)
        recall = float((top1 == np.arange(64)).mean())
        assert recall == 1.0, f"canary self-recall {recall} != 1.0"
    WorkloadManifest(
        workload="embed_faiss_index",
        started_at_unix=started,
        duration_seconds=time.time() - started,
        canary=True,
        device=device,
        config={"vram_gib": round(vram, 2), "vectors": 64, "dim": 32},
        metrics={"recall_at_1": recall, "n_queries": 64},
        timings_seconds=timings.to_dict(),
    ).write(output_dir)
    return 0


def select_model(vram_gib: float, override: str | None) -> str:
    if override:
        return override
    if vram_gib >= 16:
        return "BAAI/bge-large-en-v1.5"
    if vram_gib > 0:
        return "BAAI/bge-base-en-v1.5"
    return "sentence-transformers/all-MiniLM-L6-v2"


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
    output_dir: str = typer.Option("artifacts", help="Where to write manifest and index"),
    documents: int = typer.Option(20000, help="Number of documents to embed"),
    model_id: str = typer.Option(None, help="Override the auto-selected embedding model"),
    dataset_id: str = typer.Option("sentence-transformers/natural-questions", help="HF dataset id"),
    batch_size: int = typer.Option(256, help="Documents per encode batch"),
    probe_queries: int = typer.Option(1000, help="How many vectors to self-query for recall"),
    canary: bool = typer.Option(False, help="Sub-minute route check; no download, no GPU"),
) -> None:
    """Embed a document corpus, build a FAISS index, and verify its integrity."""
    started = time.time()
    out = resolve_output_dir(output_dir)
    if canary:
        raise typer.Exit(_run_canary(out, started))

    import faiss
    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    device, vram = detect_device()
    resolved_model = select_model(vram, model_id)
    timings = Timings()

    with timings.stage("data"):
        raw = load_dataset(dataset_id, split="train")
        raw = raw.select(range(min(documents, len(raw))))
        column = "answer" if "answer" in raw.column_names else raw.column_names[-1]
        texts = [str(row[column]) for row in raw]

    print(f"[faiss] {len(texts)} docs, model={resolved_model}, device={device}", flush=True)

    with timings.stage("load_model"):
        encoder = SentenceTransformer(resolved_model, device=device)

    chunks: list[Any] = []
    with timings.stage("embed"):
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            chunks.append(
                encoder.encode(
                    chunk,
                    batch_size=len(chunk),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )
            emit_progress(min(start + batch_size, len(texts)), len(texts))
    vectors = np.vstack(chunks).astype("float32")

    with timings.stage("index_build"):
        # Inner product on normalized vectors == cosine similarity. Flat index,
        # deliberately: an approximate index would make self-recall < 1.0 for
        # legitimate reasons and destroy the integrity signal.
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

    index_path = out / "corpus.faiss"
    with timings.stage("index_write"):
        faiss.write_index(index, str(index_path))

    with timings.stage("index_reload_verify"):
        # Round-trip through disk before verifying: this is what catches a
        # truncated or unflushed write, which is the whole point on a platform
        # whose storage layer is under test.
        reloaded = faiss.read_index(str(index_path))
        probe = vectors[: min(probe_queries, vectors.shape[0])]
        metrics = self_recall(reloaded, probe, k=5)

    metrics["index_size_mb"] = round(index_path.stat().st_size / 1024**2, 2)
    metrics["ntotal"] = int(reloaded.ntotal)
    metrics["dim"] = int(vectors.shape[1])
    embed_seconds = timings.to_dict().get("embed", 0.0) or 1.0
    metrics["docs_per_second"] = round(len(texts) / embed_seconds, 2)
    metrics["integrity_ok"] = metrics["recall_at_1"] == 1.0

    if not metrics["integrity_ok"]:
        print(
            f"[faiss] INTEGRITY-FAILURE: self-recall@1 is {metrics['recall_at_1']}, must be 1.0. "
            "The index does not contain what was written to it.",
            flush=True,
        )

    WorkloadManifest(
        workload="embed_faiss_index",
        started_at_unix=started,
        duration_seconds=time.time() - started,
        device=device,
        config={
            "model_id": resolved_model,
            "documents": len(texts),
            "batch_size": batch_size,
            "dataset_id": dataset_id,
            "vram_gib": round(vram, 2),
        },
        metrics=metrics,
        timings_seconds=timings.to_dict(),
        artifacts=[str(index_path)],
        completed_steps=len(texts),
        total_steps=len(texts),
        peak_vram_gib=peak_vram_gib(),
    ).write(out)
    print(f"[faiss] {json.dumps(metrics)}", flush=True)


if __name__ == "__main__":
    app()
