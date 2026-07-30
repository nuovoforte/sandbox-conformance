# sandbox-conformance

A conformance harness for the AI Sandbox Pilot platform. It validates the platform
against real AI workloads under both normal and failure conditions, across every
delivery route the platform exposes, and produces a pass / fail / **unobservable**
scorecard.

It is one repo, one CLI, one result schema — but deliberately not one process.
See [Why not one workload](#why-not-one-workload).

---

## The one design decision everything else follows from

**Adversarial assertions cannot live inside the sandbox.**

A job that has been OOM-killed cannot report on how its own death was handled. A
process whose filesystem is 100% full cannot reliably write a result file. A job
hung on a socket read never reaches its own reporting code at all.

So the harness is split in two halves that never share a process:

| | Runs where | Job |
|---|---|---|
| `conformance.workloads.*` | inside the sandbox | three real workloads |
| `conformance.adversarial.*` | inside the sandbox | five deliberate faults |
| `conformance.observer` | **outside**, against the REST API | asserts what the *platform* did |
| `conformance.orchestrator` | **outside** | runs the matrix, manages victim jobs |
| `conformance.report` | **outside** | scorecard |

Any harness where the failing job grades itself is measuring nothing.

## Three verdicts, not two

`pass` · `fail` · **`unobservable`** · `skipped` · `error`

`unobservable` is not a soft failure. It means the platform may well have behaved
correctly but exposes no way for anyone outside the sandbox to confirm it. On a
platform hosting 50 tenants on billable GPUs, "we think it worked" is its own
finding, and collapsing it into either `pass` or `fail` hides it. Expect the
unobservable count to be the most actionable number on your first run.

## What it covers

**Three real workloads** — real data, real tasks, each with a correctness check
that a throughput benchmark would not catch:

| Workload | Data | Correctness check |
|---|---|---|
| LoRA fine-tune | banking77 — 13,083 real bank support messages, 77 intents | JSON-validity rate and accuracy, before vs after tuning |
| Batch transcription | LibriSpeech ASR | word error rate + empty-output rate against references |
| Embedding + FAISS | natural-questions | **self-recall@1 must be exactly 1.0** |

That FAISS check is the sharpest instrument here. Querying a flat index with the
exact vectors it contains must return them — as arithmetic, not as quality. Any
deviation means a truncated write, a silent dtype downcast, an unflushed page
cache on teardown, or non-deterministic GPU reduction. All four are platform
faults; none shows up in a benchmark number.

**Five adversarial scenarios**, each declaring what the platform owes it, cited
against the Master Spec so a failure is arguable against your own document:

| Scenario | Sharpest assertion |
|---|---|
| CUDA OOM | Instance not left "nominally active" with a dead workload (spec s6) |
| Unbounded host RAM | Killed at the *container* limit, not by the host OOM killer — and a concurrent victim job survives |
| Hung network read | Something — ExpiryWorker, Gate B, bounded `hours` — eventually reclaims it |
| Kill at 60% + resume | Resumed run started from a **non-zero step**, not silently from scratch |
| Disk exhaustion | Platform retains SSH observability and can still terminate, with the disk at 100% |

**Five delivery routes**, each asked the same four questions: can it submit, can
it report status, can it retrieve the artifact, can it report an exit code.

```bash
conformance list                      # every scenario and what it asserts
conformance preflight --base-url ...  # is the platform reachable
conformance run --dry-run             # plan the matrix, launch nothing
conformance run --base-url $URL --repo-url $THIS_REPO
```

## Why not one workload

One repo, one CLI, one manifest schema: yes, and that is what this is. One
*process* running all of it: not possible, and not a stylistic preference.

A process that gets OOM-killed at scenario 1 cannot go on to run the FAISS stage.
A disk-fill scenario leaves the filesystem unusable for anything sequenced after
it. A hung-read scenario never returns. Merging them into a single job means the
first fault silently cancels every test behind it, and the run reports a single
opaque failure instead of thirteen verdicts.

What *is* unified: one repo, one `conformance` CLI, one `WorkloadManifest` schema,
one orchestrator, one scorecard.

## Matrix shape, and why it is not 3 × 5 × 2

Workload type and delivery route are orthogonal. Their cross-product is thirty
combinations, most of which re-test the same thing — Whisper-via-Jupyter and
FAISS-via-Jupyter validate the same notebook behaviour twice while burning GPU
hours. Instead:

1. **Route conformance** — a ~40-second canary through all five routes. Seconds
   of GPU time. A route that cannot carry a canary will not carry a real job, and
   you want to know that before spending an hour finding out.
2. **Real workloads** — one full run each, via the git-repo route only.
3. **Adversarial** — five scenarios, sequentially, never concurrently. Two
   simultaneous faults make blast radius unattributable.

Thirteen meaningful runs instead of thirty redundant ones.

## Corrections to stated assumptions

Both of these came out of reading Master Spec v27, and both changed the design.

**"Jupyter notebook server" is not what the platform has.** Spec section 25
defines the notebook route as an uploaded `.ipynb` executed **headlessly by
papermill** via `WorkloadRunWorker`. There is a `Jupyter` framework template, so a
container can serve Jupyter and a user can reach it through `connection_details` —
but that surface produces no `workload_runs` row, no exit code, and no audit
event. These are two different routes with two different risk profiles, so they
are tested separately. `routes.probe_interactive_jupyter` reports the interactive
surface as `unobservable` by construction: anything a developer runs there is
invisible to the platform, and therefore outside the failure detection spec
section 6 promises.

**"No runtime limit" contradicts the spec.** Section 15 lists an `ExpiryWorker`
for runtime limits, section 8's Gate B polls spend every 60s (15–20s multi-GPU)
and grace-stops at the cap, and section 8's concurrency lesson pins
`LaunchRequest.hours` to `Field(gt=0, le=720)`. Three independent backstops. The
hung-network scenario checks all three in priority order, so if none fires you
have a concrete finding rather than an assumption.

## Read this before running against a live environment

This harness creates non-terminal instances continuously, and deliberately makes
them fail — the normal operating state for a target platform's background
workers can, under some configurations, turn that into real provider calls
against a real account. That has happened before. So `conformance run` refuses
to start until you acknowledge it:

```bash
export CONFORMANCE_ACK_LIVE_CREDENTIALS=yes
```

Before that: stop every Celery worker / beat / supervisor process, and sweep
`RunPodProvider(...).list_instances()` afterwards regardless of how clean the
app-level state looks — a pod created this way has no matching DB row.

The orchestrator terminates every instance it creates in a `finally` block, but a
harness crash is still a real bill. Budget accordingly: the adversarial suite runs
two concurrent instances for the RAM and disk scenarios.

## Verification status

- 97 unit tests, ruff clean.
- Observer verdict logic tested against a fake platform in all three directions —
  including that a correct-but-invisible platform yields `unobservable`, never
  `pass`.
- All three workload canaries execute end to end (`lora`, `whisper`, `faiss`).
- `disk_fill` verified against a 200 MB tmpfs: reaches real ENOSPC, exits 28.
- `cuda_oom` aborts legibly with exit 3 where torch is absent.
- `hung_network` verified to hang rather than return.
- `host_ram` verified to allocate page-touched memory until killed.

**Not verified:** anything requiring a GPU or a live platform. No CUDA path, no
model download, no real API call has run. The endpoint paths in `platform_api.py`
come from spec section 19; the response *field names* are inferred from the
section 17 table definitions and are the most likely thing to be wrong on first
contact. They are deliberately centralized in that one module, and `pluck()`
reports a missing field as `unobservable` rather than crashing or falsely
failing — so the first run tells you which names to fix rather than dying on them.

## Layout

```
src/conformance/
  contract.py        verdict taxonomy, result schema
  platform_api.py    every endpoint path and field name, in one place
  observer.py        out-of-band assertions
  orchestrator.py    matrix execution, victim jobs, cleanup
  routes.py          five delivery routes + interactive-Jupyter probe
  report.py          HTML / JSON / text scorecard
  workloads/         lora, whisper, embed_faiss (+ shared manifest contract)
  adversarial/       five scenarios: cause + declared expectations
tests/               97 tests, no GPU or platform required
scripts/preflight.py stdlib-only environment probe
```
