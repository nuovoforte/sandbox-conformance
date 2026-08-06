"""Scenario 1 — CUDA out-of-memory.

Allocates GPU tensors until the CUDA allocator refuses. This is the single most
common way a real customer's job dies, and the Master Spec (section 6) commits to
detecting it: "In-container runtime failures (e.g. OOM after container start) ...
must be detected (health check / exit code / log pattern) and surfaced, never
left as a nominally-active dead sandbox."

That sentence is the acceptance criterion. The interesting question is not
whether the job dies — it will — but whether the platform notices, flips the
instance to RUNTIME_FAILED, terminates the provider-side resource so it stops
billing, and leaves the GPU clean for the next tenant.
"""

from __future__ import annotations

import shlex
import sys

from ..contract import Expectation
from .base import Scenario

CAUSE_SCRIPT = r"""
import sys

try:
    import torch
except ImportError as exc:
    # A sandbox whose dependency install failed must abort legibly rather than
    # emit an opaque traceback the platform will classify as a generic crash.
    print(f"SCENARIO-ABORT: torch not importable ({exc})", file=sys.stderr)
    sys.exit(3)

if not torch.cuda.is_available():
    print("SCENARIO-ABORT: no CUDA device visible", file=sys.stderr)
    sys.exit(3)

print(f"pre-alloc free/total: {torch.cuda.mem_get_info()}", flush=True)
held = []
# 512 MiB per block in fp32. Deliberately allocate without try/except around the
# whole loop: we want the process to die on the allocator's own error path, the
# way a real training job does, not to catch and exit cleanly.
block = 512 * 1024 * 1024 // 4
try:
    while True:
        held.append(torch.empty(block, dtype=torch.float32, device="cuda"))
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"allocated {allocated:.2f} GiB across {len(held)} blocks", flush=True)
except torch.cuda.OutOfMemoryError as exc:
    print(f"CUDA-OOM-REACHED: {exc}", file=sys.stderr, flush=True)
    # Exit non-zero to give the platform an exit code to classify. A real job
    # would propagate the exception; this is the same signal, deterministically.
    sys.exit(137)
"""


def cause() -> int:
    """Run the fault in-process (for local verification without the platform)."""
    namespace: dict[str, object] = {"__name__": "__cause__"}
    try:
        exec(compile(CAUSE_SCRIPT, "<cuda_oom>", "exec"), namespace)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


SCENARIO = Scenario(
    name="cuda_oom",
    summary="Allocate GPU memory until the CUDA allocator refuses",
    # NOT !r: repr() renders embedded newlines as the two literal characters
    # backslash+n, not a real newline byte. Inside single shell quotes those
    # pass through unexpanded, so `python -c` receives one "line" containing
    # literal \n sequences -- a SyntaxError before any of this script's own
    # code runs. shlex.quote() wraps in real single quotes without touching
    # embedded newlines, so the interpreter sees the actual multi-line
    # source. Found live 2026-08-06: every adversarial run's "OOM"/"kill"
    # evidence was actually this SyntaxError's own one-line source dump,
    # which happened to contain matching substrings from the script's own
    # text (e.g. "OutOfMemoryError" as a Python identifier) -- never real
    # execution.
    run_command=f"python -c {shlex.quote(CAUSE_SCRIPT)}",
    needs_gpu=True,
    needs_victim=False,
    expected_max_seconds=900.0,
    cause=cause,
    expectations=(
        Expectation(
            key="run_reaches_terminal_status",
            describes=(
                "The workload run reaches a terminal status (failed), rather than "
                "sitting in RUNNING forever after the process is gone."
            ),
            spec_ref="s25/s30 WorkloadRun staleness reaper",
        ),
        Expectation(
            key="exit_code_surfaced",
            describes="A non-zero exit code is recorded and readable via the API.",
            spec_ref="s17 workload_runs.exit_code; s18 SandboxLogType.exit_code",
        ),
        Expectation(
            key="failure_reason_is_legible",
            describes=(
                "stderr or failure_reason names the actual cause (OOM/CUDA), not a "
                "generic 'command failed'. A developer must be able to self-diagnose."
            ),
            spec_ref="s6 'surface with next-step context'",
        ),
        Expectation(
            key="instance_not_left_nominally_active",
            describes=(
                "The instance does not remain RUNNING with a dead workload. Spec "
                "section 6 forbids a 'nominally-active dead sandbox'."
            ),
            spec_ref="s6 failure handling, class 3",
        ),
        Expectation(
            key="billing_stopped_on_runtime_failure",
            describes=(
                "A RUNTIME_FAILED instance gets a provider-side terminate so it "
                "stops billing — the v27 orphaned-billing fix."
            ),
            spec_ref="s30 'RUNTIME_FAILED instances now get a best-effort terminate'",
        ),
        Expectation(
            key="gpu_reclaimed_for_next_tenant",
            describes=(
                "A fresh instance on the same GPU class starts clean, with GPU "
                "memory fully reclaimed rather than inheriting the leak."
            ),
            spec_ref="s12 benchmark reproducibility depends on a clean device",
        ),
    ),
)

if __name__ == "__main__":
    sys.exit(cause())
