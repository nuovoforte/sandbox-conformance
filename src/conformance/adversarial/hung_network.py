"""Scenario 3 — a network read that never returns.

Opens a TCP connection to a blackhole address and blocks on read with no timeout,
forever. No CPU burn, no memory growth, no GPU activity: the job looks alive to
anything watching a process table, and dead to anyone waiting for a result. This
is the hardest failure class for a platform to handle, because there is nothing
to detect — only a deadline to enforce.

You told me the platform has no runtime limit. The Master Spec disagrees, and the
spec is more likely to be right about its own code: section 15 lists an
`ExpiryWorker` whose stated job is runtime limits, section 8's Gate B polls
accumulated spend every 60s (15-20s multi-GPU) and grace-stops at the cap, and
section 8's own concurrency lesson pins `LaunchRequest.hours` to
`Field(gt=0, le=720)`.

So there are three independent backstops that should each fire on a hung job, and
this scenario checks all three in priority order. The important nuance: a GPU
sitting idle at 0% utilization while billing at full rate for hours is a *cost*
failure even when the platform eventually terminates correctly. The idle-GPU
detection expectation below is the one I expect to fail, because nothing in the
spec claims it.
"""

from __future__ import annotations

import shlex
import sys

from ..contract import Expectation
from .base import Scenario

# 203.0.113.0/24 is TEST-NET-3 (RFC 5737) — guaranteed unrouteable, so the
# connect either hangs or the read blocks indefinitely. Falling back to a real
# host would make the test dependent on someone else's uptime.
CAUSE_SCRIPT = r"""
import socket
import sys
import time

BLACKHOLE = ("203.0.113.1", 80)
print(f"connecting to blackhole {BLACKHOLE} with no timeout", flush=True)
sys.stdout.flush()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(None)  # the entire point: no timeout, ever
try:
    sock.connect(BLACKHOLE)
    print("connected; blocking on recv forever", flush=True)
    sock.recv(1)
except OSError as exc:
    # An immediate ICMP unreachable would end the test early and prove nothing,
    # so degrade into an explicit unbounded sleep, which is the same fault shape.
    print(f"connect failed ({exc}); sleeping unbounded instead", flush=True)
    while True:
        time.sleep(3600)
"""


def cause() -> int:
    namespace: dict[str, object] = {"__name__": "__cause__"}
    try:
        exec(compile(CAUSE_SCRIPT, "<hung_network>", "exec"), namespace)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


SCENARIO = Scenario(
    name="hung_network_read",
    summary="Block forever on a read from an unrouteable address",
    # NOT !r -- see cuda_oom.py's comment: repr() turns embedded newlines
    # into literal backslash-n text, which is a SyntaxError under `python -c`
    # before any of this script's own logic runs. shlex.quote() preserves
    # real newlines.
    run_command=f"python -c {shlex.quote(CAUSE_SCRIPT)}",
    needs_gpu=True,
    needs_victim=False,
    # Long by design: the whole question is whether anything ever fires. The
    # orchestrator caps this with --hang-budget so a real run stays affordable.
    expected_max_seconds=3600.0,
    cause=cause,
    expectations=(
        Expectation(
            key="something_eventually_terminates_the_job",
            describes=(
                "ExpiryWorker, Gate B, or the bounded `hours` field terminates the "
                "job. If none fires within the budget, an idle job bills forever."
            ),
            spec_ref="s15 ExpiryWorker; s8 Gate B; s8 hours=Field(gt=0, le=720)",
        ),
        Expectation(
            key="termination_reason_is_legible",
            describes=(
                "The user can tell WHY it was killed — timeout vs quota vs crash. "
                "v27 rewrote the raw 'did not reach RUNNING within N polls' message "
                "for exactly this reason."
            ),
            spec_ref="s30 frontend launch-flow message rewrite",
        ),
        Expectation(
            key="gate_b_kept_metering",
            describes=(
                "Spend kept accruing and Gate B kept polling throughout. The v27 "
                "PlatformHealthWatchdog exists because a silently-stalled Gate B "
                "disables spend-cap enforcement entirely."
            ),
            spec_ref="s30 PlatformHealthWatchdog; s8 Gate B",
        ),
        Expectation(
            key="idle_gpu_detected",
            describes=(
                "A GPU held at ~0% utilization for the whole run is flagged or "
                "surfaced. instance_metrics records gpu_utilization_pct_avg, but "
                "nothing in the spec acts on it — expected to be a real gap."
            ),
            spec_ref="s26 instance_metrics.gpu_utilization_pct_avg (no consumer)",
        ),
        Expectation(
            key="run_reaches_terminal_status",
            describes="The workload run does not remain RUNNING after the instance ends.",
            spec_ref="s30 WorkloadRun staleness reaper",
        ),
    ),
)

if __name__ == "__main__":
    sys.exit(cause())
