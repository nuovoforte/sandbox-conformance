"""Scenario 4 — kill training at 60%, resume from checkpoint.

The only scenario here that tests a *positive* capability rather than graceful
degradation, and the most commercially important one: a fine-tune that cannot
survive an interruption is a fine-tune nobody will run on your platform for real
money.

Design, and why it is shaped this way:

  1. Launch with a persistent volume attached (spec section 10 / section 28's
     create-and-attach-at-launch). Checkpoints must land on the volume, not the
     container filesystem, or there is nothing to resume from by definition.
  2. Run the LoRA workload with step-based checkpointing and watch progress.
  3. At ~60% of total steps, terminate the instance through the platform's own
     user-facing endpoint (spec section 26) — not by killing the process over
     SSH. The point is to test the platform's termination path, which is what a
     real quota exhaustion or grace-stop does.
  4. Relaunch against the SAME volume and resume.
  5. Compare final metrics against a known-good uninterrupted baseline.

Step 5 is the assertion most harnesses omit and the one that catches the worst
class of bug. A resume that "works" but silently restarts from step 0, or loads a
truncated optimizer state and converges worse, passes every check except this one.
A tolerance band rather than exact equality, because non-determinism in dataloader
ordering after a resume is expected and acceptable.

There is one platform-shaped sharp edge worth naming: spec section 29 says
cross-provider migration drops the provider-pinned volume, and section 28 says
no provider supports live-attaching storage to a running pod. So resume only works
if the relaunch targets the same provider. If your grace-stop path ever relaunches
elsewhere, checkpoints are lost — worth checking separately.
"""

from __future__ import annotations

from ..contract import Expectation
from .base import Scenario

# The kill happens from outside, so the in-sandbox command is just the real
# workload with checkpointing turned up. `--kill-marker` makes the 60% point
# observable in stdout so the orchestrator knows when to pull the trigger
# instead of guessing at a wall-clock fraction.
RUN_COMMAND = (
    "python -m conformance.workloads.lora run "
    "--output-dir /mnt/volume/artifacts "
    "--checkpoint-every-steps 10 "
    "--emit-progress "
    "--skip-baseline"
)

RESUME_COMMAND = (
    "python -m conformance.workloads.lora run "
    "--output-dir /mnt/volume/artifacts "
    "--checkpoint-every-steps 10 "
    "--emit-progress "
    "--skip-baseline "
    "--resume-from /mnt/volume/artifacts/training"
)

SCENARIO = Scenario(
    name="kill_at_60_resume_from_checkpoint",
    summary="Terminate a training run at ~60% of steps, then resume on the same volume",
    run_command=RUN_COMMAND,
    needs_gpu=True,
    needs_victim=False,
    requires_volume=True,
    expected_max_seconds=3600.0,
    expectations=(
        Expectation(
            key="checkpoint_written_before_kill",
            describes=(
                "At least one step-based checkpoint reached the persistent volume "
                "before termination. Without this the rest is untestable."
            ),
            spec_ref="s10 persistent volumes",
        ),
        Expectation(
            key="checkpoint_survived_termination",
            describes=(
                "The checkpoint is still readable on the volume after the instance "
                "is gone. Tests that volume lifetime is genuinely independent of "
                "instance lifetime."
            ),
            spec_ref="s10 volumes create/attach/reuse; s28 relaunch-with-volume",
        ),
        Expectation(
            key="relaunch_reattached_same_volume",
            describes=(
                "The new instance mounted the same volume with data intact, via "
                "relaunch-with-volume or an explicit volume_id at launch."
            ),
            spec_ref="s28 POST /sandbox/instances/{id}/relaunch-with-volume",
        ),
        Expectation(
            key="resumed_from_nonzero_step",
            describes=(
                "The resumed run started from the checkpointed step, not from zero. "
                "A silent restart-from-scratch is the most common false pass here."
            ),
            spec_ref="workload contract: manifest records resumed_from_step",
        ),
        Expectation(
            key="resumed_metrics_match_baseline",
            describes=(
                "Final accuracy and JSON-validity land within tolerance of an "
                "uninterrupted baseline run. Catches a resume that loads truncated "
                "optimizer state and quietly converges worse."
            ),
            spec_ref="workload contract: manifest metrics comparison",
        ),
        Expectation(
            key="usage_metered_across_both_instances",
            describes=(
                "Both instances' GPU time appears in the usage ledger. The v26 fix "
                "closed a real accounting gap where a user-initiated terminate "
                "between polls silently lost the final stretch of usage."
            ),
            spec_ref="s29 usage-tracking bug fix; s17 usage_records",
        ),
        Expectation(
            key="lineage_is_reconstructable",
            describes=(
                "An auditor can tell the two runs were one logical job. Spec section "
                "18 admits relaunch/migrate 'compose already-audited primitives but "
                "don't tag the pair as one linked event' — a known gap this asserts."
            ),
            spec_ref="s18 AuditAction known gap; s17 workload_runs.rerun_of_id",
        ),
    ),
)
