"""Out-of-band assertions about platform behaviour.

This module never runs inside the sandbox. It talks only to the platform's REST
API, because the entire premise is that a job which has been OOM-killed, hung, or
had its filesystem filled cannot report on how it was handled.

The three-way verdict is the load-bearing design choice. Most harnesses collapse
"the platform did the wrong thing" and "I cannot tell what the platform did" into
one FAIL, which is wrong in both directions: it slanders correct behaviour that is
merely invisible, and it lets a genuine observability hole hide behind a fixed
bug. Here they stay separate, and UNOBSERVABLE is reported as its own category on
the scorecard.
"""

from __future__ import annotations

import re
from typing import Any

from .contract import Observation, Verdict
from .platform_api import (
    DEAD_INSTANCE_STATES,
    F_DISK_TOTAL,
    F_DISK_USED,
    F_EXIT_CODE,
    F_FAILURE_REASON,
    F_GPU_UTIL,
    F_RAM_USED,
    F_RUN_STATUS,
    F_STDERR,
    F_STDOUT,
    TERMINAL_RUN_STATES,
    PlatformClient,
    PlatformError,
    pluck,
)

# Patterns that make a failure self-diagnosable to a developer reading it cold.
OOM_PATTERNS = re.compile(
    r"out of memory|outofmemory|cuda.{0,20}oom|cuda error|memory allocat", re.IGNORECASE
)
ENOSPC_PATTERNS = re.compile(
    r"no space left|enospc|errno 28|disk (?:is )?full|quota exceeded", re.IGNORECASE
)
TIMEOUT_PATTERNS = re.compile(
    r"timeout|timed out|runtime limit|expired|exceeded.{0,20}(?:hours|duration)|"
    r"quota|grace.?stop|max runtime",
    re.IGNORECASE,
)
CGROUP_KILL_PATTERNS = re.compile(
    r"killed|sigkill|exit(?:ed)? 137|oom.?kill|memory cgroup", re.IGNORECASE
)


def _text_of(run: dict[str, Any] | None) -> str:
    """Concatenate every human-readable field on a workload run."""
    if not run:
        return ""
    parts: list[str] = []
    for candidates in (F_STDERR, F_STDOUT, F_FAILURE_REASON):
        found = pluck(run, candidates)
        if found.value:
            parts.append(str(found.value))
    return "\n".join(parts)


class Observer:
    """Checks one scenario's expectations against the platform's own API."""

    def __init__(self, client: PlatformClient) -> None:
        self.client = client

    # --- generic checks reused across scenarios ----------------------------

    def run_reaches_terminal_status(self, run: dict[str, Any] | None) -> Observation:
        key = "run_reaches_terminal_status"
        if run is None:
            return Observation(
                key,
                Verdict.UNOBSERVABLE,
                "No workload run was retrievable for this instance at all.",
            )
        status = pluck(run, F_RUN_STATUS)
        if not status.observable:
            return Observation(
                key,
                Verdict.UNOBSERVABLE,
                f"No status field found among {F_RUN_STATUS} on the workload run.",
                {"run_keys": sorted(run)},
            )
        value = str(status.value).lower()
        if value in TERMINAL_RUN_STATES:
            return Observation(
                key, Verdict.PASS, f"Run reached terminal status {value!r}.", {"status": value}
            )
        return Observation(
            key,
            Verdict.FAIL,
            f"Run still {value!r} after the fault and the wait budget — the staleness "
            "reaper did not fire, so this run is stuck forever.",
            {"status": value},
        )

    def exit_code_surfaced(
        self, run: dict[str, Any] | None, expect: int | None = None
    ) -> Observation:
        key = "exit_code_surfaced"
        found = pluck(run, F_EXIT_CODE)
        if not found.observable:
            return Observation(
                key,
                Verdict.UNOBSERVABLE,
                f"No exit-code field among {F_EXIT_CODE}; cannot tell how the process died.",
                {"run_keys": sorted(run) if run else []},
            )
        if found.value is None:
            return Observation(
                key,
                Verdict.FAIL,
                f"Field {found.found_as!r} is present but null on a finished run — "
                "the platform recorded no exit code for a process that definitely exited.",
            )
        code = int(found.value)
        if code == 0:
            return Observation(
                key,
                Verdict.FAIL,
                "Exit code 0 for a run that was supposed to fail. Either the fault "
                "never triggered or the platform lost the real status.",
                {"exit_code": 0},
            )
        if expect is not None and code != expect:
            return Observation(
                key,
                Verdict.PASS,
                f"Non-zero exit code {code} recorded (expected {expect}; a different "
                "non-zero code still proves the failure path works).",
                {"exit_code": code, "expected": expect},
            )
        return Observation(key, Verdict.PASS, f"Exit code {code} recorded.", {"exit_code": code})

    def message_matches(
        self,
        key: str,
        run: dict[str, Any] | None,
        pattern: re.Pattern[str],
        what: str,
    ) -> Observation:
        text = _text_of(run)
        if not text:
            return Observation(
                key,
                Verdict.UNOBSERVABLE,
                "No stdout, stderr, or failure_reason content was retrievable, so "
                "legibility cannot be judged.",
            )
        match = pattern.search(text)
        if match:
            return Observation(
                key,
                Verdict.PASS,
                f"Output names {what} ({match.group(0)!r}).",
                {"matched": match.group(0)},
            )
        return Observation(
            key,
            Verdict.FAIL,
            f"Output never mentions {what}. A developer reading this cannot "
            "self-diagnose; they will open a support ticket.",
            {"tail": text[-400:]},
        )

    def instance_not_left_nominally_active(self, instance_id: str) -> Observation:
        key = "instance_not_left_nominally_active"
        try:
            found = self.client.instance_status(instance_id)
        except PlatformError as exc:
            return Observation(key, Verdict.ERROR, f"Could not read instance status: {exc}")
        if not found.observable:
            return Observation(
                key, Verdict.UNOBSERVABLE, f"No status field among {F_INSTANCE_STATUS_HINT}."
            )
        state = str(found.value).lower()
        if state == "running":
            return Observation(
                key,
                Verdict.FAIL,
                "Instance still RUNNING after its workload died. Spec section 6 "
                "explicitly forbids a nominally-active dead sandbox — this is "
                "billing with nothing watching it.",
                {"status": state},
            )
        return Observation(key, Verdict.PASS, f"Instance moved to {state!r}.", {"status": state})

    def billing_stopped(self, instance_id: str) -> Observation:
        key = "billing_stopped_on_runtime_failure"
        try:
            found = self.client.instance_status(instance_id)
        except PlatformError as exc:
            return Observation(key, Verdict.ERROR, f"Could not read instance status: {exc}")
        state = str(found.value).lower() if found.value else ""
        if state in DEAD_INSTANCE_STATES:
            return Observation(
                key,
                Verdict.PASS,
                f"Instance reached {state!r}, so the provider-side resource was released.",
                {"status": state},
            )
        if state == "runtime_failed":
            return Observation(
                key,
                Verdict.UNOBSERVABLE,
                "Instance is RUNTIME_FAILED. The v27 fix should have issued a "
                "provider-side terminate, but that is only confirmable against the "
                "provider's own API — check RunPod's list_instances directly.",
                {"status": state},
            )
        return Observation(
            key,
            Verdict.FAIL,
            f"Instance in {state!r} — no evidence the billable resource was released.",
            {"status": state},
        )

    # --- metrics-based checks ---------------------------------------------

    def metrics_kept_reporting(
        self, instance_id: str, field: tuple[str, ...], label: str, min_samples: int = 2
    ) -> Observation:
        key = f"metrics_kept_reporting_{label}" if label != "ram" else "metrics_kept_reporting"
        series = self.client.metrics_series(instance_id)
        if not series:
            return Observation(
                key,
                Verdict.UNOBSERVABLE,
                "No instance_metrics rows retrievable. Either the poll never ran or "
                "retention already pruned them — both make this scenario blind.",
            )
        values = [pluck(row, field).value for row in series]
        present = [v for v in values if v is not None]
        if len(present) < min_samples:
            return Observation(
                key,
                Verdict.FAIL,
                f"Only {len(present)} usable {label} sample(s) across {len(series)} "
                "metric rows — the metrics path died during the incident it exists "
                "to report on.",
                {"rows": len(series), "usable": len(present)},
            )
        return Observation(
            key,
            Verdict.PASS,
            f"{len(present)} {label} samples recorded across the run.",
            {"samples": len(present), "first": present[0], "last": present[-1]},
        )

    def disk_metrics_showed_ramp(self, instance_id: str) -> Observation:
        key = "disk_metrics_showed_the_ramp"
        series = self.client.metrics_series(instance_id)
        if not series:
            return Observation(
                key, Verdict.UNOBSERVABLE, "No metric rows retrievable for this instance."
            )
        used = [pluck(r, F_DISK_USED).value for r in series]
        total = [pluck(r, F_DISK_TOTAL).value for r in series]
        usable = [(u, t) for u, t in zip(used, total, strict=False) if u is not None and t]
        if len(usable) < 2:
            return Observation(
                key,
                Verdict.UNOBSERVABLE,
                f"Fewer than two rows carried both {F_DISK_USED[0]} and {F_DISK_TOTAL[0]}.",
                {"rows": len(series)},
            )
        first_pct = 100 * usable[0][0] / usable[0][1]
        last_pct = 100 * usable[-1][0] / usable[-1][1]
        if last_pct > first_pct + 5:
            return Observation(
                key,
                Verdict.PASS,
                f"Disk usage climbed {first_pct:.1f}% -> {last_pct:.1f}%, so this is "
                "usable as an early-warning signal.",
                {"first_pct": round(first_pct, 1), "last_pct": round(last_pct, 1)},
            )
        return Observation(
            key,
            Verdict.FAIL,
            f"Disk usage barely moved ({first_pct:.1f}% -> {last_pct:.1f}%) while the "
            "job filled the filesystem — the metric does not reflect reality.",
            {"first_pct": round(first_pct, 1), "last_pct": round(last_pct, 1)},
        )

    def idle_gpu_detected(self, instance_id: str, threshold: float = 5.0) -> Observation:
        key = "idle_gpu_detected"
        series = self.client.metrics_series(instance_id)
        utils = [pluck(r, F_GPU_UTIL).value for r in series]
        present = [float(u) for u in utils if u is not None]
        if not present:
            return Observation(
                key,
                Verdict.UNOBSERVABLE,
                "No GPU-utilization samples retrievable, so idleness is invisible.",
            )
        peak = max(present)
        if peak > threshold:
            return Observation(
                key,
                Verdict.PASS,
                f"GPU reached {peak:.1f}% utilization, so the job was not idle.",
                {"peak_util": peak},
            )
        return Observation(
            key,
            Verdict.FAIL,
            f"GPU never exceeded {peak:.1f}% across {len(present)} samples while "
            "billing at full rate, and nothing flagged it. The data to detect this "
            "is collected but unused — an idle-GPU alert is a cheap, high-value add.",
            {"peak_util": peak, "samples": len(present)},
        )

    def platform_retained_ssh_observability(self, instance_id: str) -> Observation:
        key = "platform_retained_ssh_observability"
        try:
            latest = self.client.metrics_latest(instance_id)
        except PlatformError as exc:
            return Observation(key, Verdict.FAIL, f"Metrics endpoint errored: {exc}")
        if latest is None:
            return Observation(
                key,
                Verdict.FAIL,
                "No latest-metrics snapshot available while the fault was active. The "
                "SSH control path that carries dispatch, metrics, benchmarks, and "
                "SFTP appears to have gone down with the sandbox.",
            )
        return Observation(
            key,
            Verdict.PASS,
            "Metrics snapshot still served during the fault, so the SSH control path held.",
            {"fields": sorted(latest)[:12]},
        )

    def terminate_still_works(self, instance_id: str) -> Observation:
        key = "terminate_still_works"
        try:
            self.client.terminate(instance_id)
        except PlatformError as exc:
            return Observation(
                key,
                Verdict.FAIL,
                f"Terminate was rejected while the fault was active: {exc}. An "
                "instance nobody can stop bills until a human notices.",
            )
        state, waited = self.client.wait_for_instance_state(
            instance_id, DEAD_INSTANCE_STATES, timeout=300, poll=10
        )
        if state in DEAD_INSTANCE_STATES:
            return Observation(
                key,
                Verdict.PASS,
                f"Instance terminated in {waited:.0f}s despite the fault.",
                {"final_status": state, "seconds": round(waited, 1)},
            )
        return Observation(
            key,
            Verdict.FAIL,
            f"Terminate accepted but instance sat in {state!r} for {waited:.0f}s "
            "without reaching a dead state.",
            {"final_status": state},
        )

    # --- victim / blast radius --------------------------------------------

    def victim_survived(self, victim_run: dict[str, Any] | None) -> Observation:
        key = "victim_job_survived"
        if victim_run is None:
            return Observation(
                key,
                Verdict.UNOBSERVABLE,
                "No victim run retrievable, so blast radius is unknown. This is the "
                "single most important assertion for a 50-tenant platform.",
            )
        status = pluck(victim_run, F_RUN_STATUS)
        code = pluck(victim_run, F_EXIT_CODE)
        state = str(status.value).lower() if status.value else "unknown"
        if state == "succeeded" or (code.value is not None and int(code.value) == 0):
            return Observation(
                key,
                Verdict.PASS,
                "Healthy concurrent job completed normally — the fault stayed inside "
                "its own container.",
                {"victim_status": state},
            )
        return Observation(
            key,
            Verdict.FAIL,
            f"Concurrent healthy job ended {state!r}. One tenant's fault destroyed "
            "another tenant's work; this is an isolation failure, not a workload bug.",
            {
                "victim_status": state,
                "victim_exit": code.value,
                "tail": _text_of(victim_run)[-300:],
            },
        )


# Kept out of the import block above because it is only used in an error string.
F_INSTANCE_STATUS_HINT = ("status", "instance_status")

__all__ = [
    "CGROUP_KILL_PATTERNS",
    "ENOSPC_PATTERNS",
    "F_RAM_USED",
    "OOM_PATTERNS",
    "TIMEOUT_PATTERNS",
    "Observer",
]
