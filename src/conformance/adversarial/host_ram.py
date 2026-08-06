"""Scenario 2 — unbounded host RAM allocation.

Grows a Python list of bytearrays until something kills the process. The question
is *what* kills it, and the difference matters enormously:

  - The container's cgroup limit kills it  ->  correct. One tenant's mistake is
    bounded to one tenant.
  - The host OOM killer kills it           ->  wrong. The kernel picks a victim
    by heuristic score, which may well be someone else's job, the SSH daemon the
    platform depends on for metrics and workload dispatch, or the container
    runtime itself.

Because the second case is invisible from the attacking job's perspective, this
scenario runs a victim job concurrently on the same node. If the victim dies, the
sandbox is not isolating memory and the platform cannot safely host 50 tenants.

Secondary target: the Master Spec's metrics path (section 26) polls RAM over SSH.
Under host memory pressure that SSH round trip is exactly what fails first, so
this scenario also tests whether observability survives the incident it is
supposed to be reporting on.
"""

from __future__ import annotations

import shlex
import sys

from ..contract import Expectation
from .base import Scenario

CAUSE_SCRIPT = r"""
import os
import sys
import time

CHUNK = 256 * 1024 * 1024  # 256 MiB
held = []
total = 0
# Touch every page: a bytearray is lazily backed on some allocators, and an
# untouched allocation does not create real memory pressure.
while True:
    block = bytearray(CHUNK)
    for offset in range(0, CHUNK, 4096):
        block[offset] = 1
    held.append(block)
    total += CHUNK
    print(f"held {total / 1024**3:.2f} GiB rss-backed", flush=True)
    # Deliberately slow enough that a 15-60s metrics poll can observe the ramp.
    time.sleep(0.5)
    if total > 2048 * 1024**3:
        print("SCENARIO-ABORT: 2 TiB allocated without a kill", file=sys.stderr)
        os._exit(4)
"""

VICTIM_SCRIPT = r"""
import sys
import time

# A deliberately boring healthy tenant: writes a heartbeat, exits 0 if it
# survives the window. Its survival IS the blast-radius assertion.
DEADLINE = 900
start = time.monotonic()
beats = 0
while time.monotonic() - start < DEADLINE:
    beats += 1
    print(f"victim-heartbeat {beats}", flush=True)
    time.sleep(10)
print(f"VICTIM-SURVIVED beats={beats}", flush=True)
sys.exit(0)
"""


def cause() -> int:
    namespace: dict[str, object] = {"__name__": "__cause__"}
    try:
        exec(compile(CAUSE_SCRIPT, "<host_ram>", "exec"), namespace)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


SCENARIO = Scenario(
    name="host_ram_exhaustion",
    summary="Allocate host RAM without bound until something kills the process",
    # NOT !r -- see cuda_oom.py's comment: repr() turns embedded newlines
    # into literal backslash-n text, which is a SyntaxError under `python -c`
    # before any of this script's own logic runs. shlex.quote() preserves
    # real newlines.
    run_command=f"python -c {shlex.quote(CAUSE_SCRIPT)}",
    needs_gpu=False,
    needs_victim=True,
    expected_max_seconds=1200.0,
    cause=cause,
    expectations=(
        Expectation(
            key="run_reaches_terminal_status",
            describes="The run is marked failed rather than stuck RUNNING after the kill.",
            spec_ref="s30 WorkloadRun staleness reaper",
        ),
        Expectation(
            key="killed_at_container_limit",
            describes=(
                "Termination is attributable to the container's own memory limit "
                "(exit 137 / cgroup OOM), not to a host-level kill of an arbitrary "
                "victim process."
            ),
            spec_ref="s6 in-container runtime failure detection",
        ),
        Expectation(
            key="victim_job_survived",
            describes=(
                "A healthy concurrent job on the same node completes normally. If it "
                "does not, one tenant can take out another's work."
            ),
            spec_ref="s1 '50 real developers' multi-tenant premise",
            blast_radius=True,
        ),
        Expectation(
            key="metrics_kept_reporting",
            describes=(
                "instance_metrics kept recording RAM during the ramp. If the SSH "
                "metrics poll died first, the platform is blind during exactly the "
                "incident it needs to report."
            ),
            spec_ref="s26 MetricsPollWorker over SSH",
        ),
        Expectation(
            key="instance_not_left_nominally_active",
            describes="The instance is not left RUNNING with a dead workload.",
            spec_ref="s6 failure handling, class 3",
        ),
    ),
)

if __name__ == "__main__":
    sys.exit(cause())
