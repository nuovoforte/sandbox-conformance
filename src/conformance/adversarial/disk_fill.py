"""Scenario 5 — fill the disk completely during execution.

Writes incompressible random bytes until the filesystem returns ENOSPC. Two
distinct failure surfaces, and the second is the one that actually hurts:

  1. The job's own writes fail. Expected, and fine if the error is legible.
  2. Everything the PLATFORM does over that filesystem fails. Per the Master
     Spec, the sandbox's SSH channel carries workload dispatch (section 25),
     metrics polling (section 26), benchmark scripts (section 12), and SFTP file
     upload (section 29). A full disk breaks sshd's ability to write session
     state, so the platform can lose the ability to observe or terminate the very
     instance that is misbehaving.

That second surface is why this scenario runs a victim job: if a shared
filesystem is involved rather than a per-container overlay, one tenant filling
disk stops another tenant's job from writing checkpoints.

Deliberately writes from /dev/urandom rather than zeros — a filesystem with
compression or sparse-file handling will happily absorb terabytes of zeros and
report the test as passing when nothing was ever filled.
"""

from __future__ import annotations

import shlex
import sys

from ..contract import Expectation
from .base import Scenario

CAUSE_SCRIPT = r"""
import os
import shutil
import sys

TARGET = os.environ.get("FILL_TARGET", "/tmp/fill")
CHUNK = 64 * 1024 * 1024

os.makedirs(TARGET, exist_ok=True)
total, used, free = shutil.disk_usage(TARGET)
print(f"start: {free / 1024**3:.2f} GiB free of {total / 1024**3:.2f} GiB", flush=True)

# Incompressible payload: zeros are absorbed by compression/sparse handling and
# would let this test pass without ever filling anything.
payload = os.urandom(CHUNK)
written = 0
index = 0

while True:
    path = os.path.join(TARGET, f"ballast_{index:05d}.bin")
    try:
        with open(path, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())  # defeat page cache; force real allocation
    except OSError as exc:
        if exc.errno == 28:  # ENOSPC
            print(f"DISK-FULL-REACHED after {written / 1024**3:.2f} GiB: {exc}",
                  file=sys.stderr, flush=True)
            # Leave the disk full. Cleaning up here would hide whether the
            # PLATFORM can reclaim the volume on teardown, which is the point.
            sys.exit(28)
        print(f"unexpected OSError: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
    written += CHUNK
    index += 1
    if index % 8 == 0:
        _, _, free = shutil.disk_usage(TARGET)
        print(f"wrote {written / 1024**3:.2f} GiB, {free / 1024**3:.2f} GiB free",
              flush=True)
"""


def cause() -> int:
    namespace: dict[str, object] = {"__name__": "__cause__"}
    try:
        exec(compile(CAUSE_SCRIPT, "<disk_fill>", "exec"), namespace)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


SCENARIO = Scenario(
    name="disk_exhaustion",
    summary="Write incompressible data until the filesystem returns ENOSPC",
    # NOT !r -- see cuda_oom.py's comment: repr() turns embedded newlines
    # into literal backslash-n text, which is a SyntaxError under `python -c`
    # before any of this script's own logic runs. shlex.quote() preserves
    # real newlines.
    run_command=f"python -c {shlex.quote(CAUSE_SCRIPT)}",
    needs_gpu=False,
    needs_victim=True,
    expected_max_seconds=1800.0,
    cause=cause,
    expectations=(
        Expectation(
            key="run_reaches_terminal_status",
            describes="The run is marked failed, with ENOSPC reflected in the exit code.",
            spec_ref="s17 workload_runs.exit_code; s30 staleness reaper",
        ),
        Expectation(
            key="enospc_error_is_legible",
            describes=(
                "stderr or failure_reason says the disk filled. 'Command exited 1' "
                "sends a developer hunting in the wrong place for an hour."
            ),
            spec_ref="s6 'surface with next-step context'",
        ),
        Expectation(
            key="platform_retained_ssh_observability",
            describes=(
                "Metrics polling and status checks still worked with the disk at "
                "100%. The whole control path runs over this instance's SSH — if it "
                "breaks, the platform is blind precisely when it must act."
            ),
            spec_ref="s26 MetricsPollWorker; s25 WorkloadRunWorker over SSH",
        ),
        Expectation(
            key="disk_metrics_showed_the_ramp",
            describes=(
                "instance_metrics.disk_used_gb tracked toward total. The field is "
                "collected; this checks it is actually usable as an early warning."
            ),
            spec_ref="s17 instance_metrics.disk_used_gb/disk_total_gb",
        ),
        Expectation(
            key="terminate_still_works",
            describes=(
                "The platform can still terminate the instance with the disk full. "
                "An unterminable instance bills until someone notices by hand."
            ),
            spec_ref="s26 user-facing terminate; s15 CleanupWorker",
        ),
        Expectation(
            key="volume_reclaimed_on_teardown",
            describes=(
                "Storage is actually released after termination. Spec section 8 "
                "notes storage keeps accruing during grace-stop, so a leaked full "
                "volume is a standing bill."
            ),
            spec_ref="s8 storage accrual; s15 VolumeReaperWorker",
        ),
        Expectation(
            key="victim_job_survived",
            describes=(
                "A concurrent job kept writing successfully. If it failed too, the "
                "filesystem is shared across tenants."
            ),
            spec_ref="s1 multi-tenant premise",
            blast_radius=True,
        ),
    ),
)

if __name__ == "__main__":
    sys.exit(cause())
