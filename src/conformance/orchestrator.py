"""Runs the matrix and collects verdicts.

Sequencing rules that matter:

  - Route conformance runs FIRST, with canary workloads. A route that cannot
    submit a 40-second job will not carry a 40-minute one, and discovering that
    after burning GPU hours is avoidable.
  - Adversarial scenarios run LAST and one at a time. Two concurrent faults on
    one node make blast radius unattributable — if a victim dies you cannot say
    which attacker killed it.
  - Every scenario terminates its own instances in a finally block. On a platform
    where quota exhaustion is the only other backstop, a leaked instance from a
    crashed harness is a real bill.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .adversarial import BY_NAME
from .contract import ConformanceReport, Observation, Route, ScenarioResult, Verdict
from .observer import (
    CGROUP_KILL_PATTERNS,
    ENOSPC_PATTERNS,
    OOM_PATTERNS,
    TIMEOUT_PATTERNS,
    Observer,
)
from .platform_api import DEAD_INSTANCE_STATES, F_RAM_USED, PlatformError, pluck
from .routes import RouteRunner, check_route, probe_interactive_jupyter
from .workloads import run_command
from .workloads.base import PROGRESS_MARKER

if TYPE_CHECKING:
    from .adversarial.base import Scenario
    from .platform_api import PlatformClient

VICTIM_COMMAND = run_command("lora", canary=False, extra="--train-rows 400 --skip-baseline")


@dataclass(slots=True)
class RunOptions:
    repo_url: str
    repo_ref: str = "main"
    template: str = "pytorch"
    gpu: str | None = None
    hang_budget_seconds: float = 1800.0
    dry_run: bool = False
    skip_adversarial: bool = False
    skip_workloads: bool = False
    scenarios: tuple[str, ...] = ()


class Orchestrator:
    def __init__(self, client: PlatformClient, options: RunOptions) -> None:
        self.client = client
        self.options = options
        self.observer = Observer(client)
        self.runner = RouteRunner(
            client,
            repo_url=options.repo_url,
            repo_ref=options.repo_ref,
            template=options.template,
            gpu=options.gpu,
        )

    # --- top level ---------------------------------------------------------

    def run_all(self) -> ConformanceReport:
        report = ConformanceReport(
            platform_base_url=self.client.base_url, started_at_unix=time.time()
        )
        started = time.monotonic()

        for result in self.run_route_conformance():
            report.results.append(result)

        if self.options.skip_workloads:
            report.skipped.append("real workloads (--skip-workloads)")
        else:
            for result in self.run_real_workloads():
                report.results.append(result)

        if self.options.skip_adversarial:
            report.skipped.append("adversarial suite (--skip-adversarial)")
        else:
            for result in self.run_adversarial():
                report.results.append(result)

        report.duration_seconds = time.monotonic() - started
        return report

    # --- phase 1: routes ---------------------------------------------------

    def run_route_conformance(self) -> list[ScenarioResult]:
        """Cheap canary through every route. Seconds of GPU time, not hours."""
        results: list[ScenarioResult] = []
        canary = run_command("lora", canary=True)

        plans: list[tuple[Route, Any]] = [
            (Route.GIT_REPO, lambda: self.runner.via_git_repo(canary)),
            (Route.REST_API, lambda: self.runner.via_rest_api(canary)),
            (Route.NOTEBOOK, lambda: self.runner.via_notebook("lora", canary=True)),
            (Route.CLI, lambda: self.runner.via_cli(canary)),
            (Route.SDK, lambda: self.runner.via_sdk(canary)),
        ]

        for route, launch_fn in plans:
            started = time.time()
            if self.options.dry_run:
                results.append(
                    ScenarioResult(
                        scenario=f"route_canary::{route.value}",
                        route=route,
                        instance_id=None,
                        workload_run_id=None,
                        started_at_unix=started,
                        duration_seconds=0.0,
                        observations=[
                            Observation(f"{route.value}.can_submit", Verdict.SKIPPED, "dry run")
                        ],
                    )
                )
                continue

            launch = launch_fn()
            observations = check_route(self.client, launch, timeout=900)
            if launch.instance_id:
                observations.append(probe_interactive_jupyter(self.client, launch.instance_id))
            self._cleanup(launch.instance_id)
            results.append(
                ScenarioResult(
                    scenario=f"route_canary::{route.value}",
                    route=route,
                    instance_id=launch.instance_id,
                    workload_run_id=launch.workload_run_id,
                    started_at_unix=started,
                    duration_seconds=time.time() - started,
                    observations=observations,
                )
            )
        return results

    # --- phase 2: real workloads -------------------------------------------

    def run_real_workloads(self) -> list[ScenarioResult]:
        """One full run of each real workload, through the git_repo route only."""
        results: list[ScenarioResult] = []
        for workload in ("lora", "whisper", "faiss"):
            started = time.time()
            if self.options.dry_run:
                results.append(self._dry(f"workload::{workload}", Route.GIT_REPO, started))
                continue

            command = run_command(workload)
            launch = self.runner.via_git_repo(command)
            observations = check_route(self.client, launch, timeout=7200)
            metrics = None
            if launch.instance_id:
                run = self.client.latest_workload_run(launch.instance_id)
                metrics = _manifest_from_stdout(run)
                observations.append(self._integrity_check(workload, metrics))
            self._cleanup(launch.instance_id)
            results.append(
                ScenarioResult(
                    scenario=f"workload::{workload}",
                    route=Route.GIT_REPO,
                    instance_id=launch.instance_id,
                    workload_run_id=launch.workload_run_id,
                    started_at_unix=started,
                    duration_seconds=time.time() - started,
                    observations=observations,
                    workload_metrics=metrics,
                )
            )
        return results

    def _integrity_check(self, workload: str, manifest: dict[str, Any] | None) -> Observation:
        """Did the workload produce a correct result, not merely a finished one?"""
        key = f"{workload}.result_is_correct"
        if not manifest:
            return Observation(
                key, Verdict.UNOBSERVABLE, "No workload manifest recoverable from run output."
            )
        metrics = manifest.get("metrics") or {}
        if workload == "faiss":
            recall = metrics.get("recall_at_1")
            if recall is None:
                return Observation(key, Verdict.UNOBSERVABLE, "No recall_at_1 in manifest.")
            if recall == 1.0:
                return Observation(key, Verdict.PASS, "FAISS self-recall@1 is exactly 1.0.")
            return Observation(
                key,
                Verdict.FAIL,
                f"FAISS self-recall@1 is {recall}, must be exactly 1.0. The index does "
                "not contain what was written to it — a storage or determinism fault.",
                {"recall_at_1": recall},
            )
        if workload == "whisper":
            wer = metrics.get("wer")
            empty = metrics.get("empty_rate", 0)
            if wer is None:
                return Observation(key, Verdict.UNOBSERVABLE, "No WER in manifest.")
            if wer < 0.25 and empty < 0.05:
                return Observation(key, Verdict.PASS, f"WER {wer} with {empty:.1%} empty outputs.")
            return Observation(
                key,
                Verdict.FAIL,
                f"WER {wer} / empty rate {empty:.1%} — the job ran but transcribed badly, "
                "which a throughput benchmark would have reported as success.",
                {"wer": wer, "empty_rate": empty},
            )
        tuned = (metrics or {}).get("tuned") or {}
        validity = tuned.get("json_valid_rate")
        if validity is None:
            return Observation(key, Verdict.UNOBSERVABLE, "No tuned metrics in manifest.")
        if validity >= 0.9:
            return Observation(key, Verdict.PASS, f"JSON-validity {validity} after tuning.")
        return Observation(
            key,
            Verdict.FAIL,
            f"JSON-validity only {validity} after tuning — training completed but did "
            "not learn, which points at a silently broken compute path.",
            {"json_valid_rate": validity},
        )

    # --- phase 3: adversarial ----------------------------------------------

    def run_adversarial(self) -> list[ScenarioResult]:
        names = self.options.scenarios or tuple(BY_NAME)
        results: list[ScenarioResult] = []
        for name in names:
            scenario = BY_NAME.get(name)
            if scenario is None:
                continue
            started = time.time()
            if self.options.dry_run:
                results.append(self._dry(f"adversarial::{name}", Route.GIT_REPO, started))
                continue
            if name == "kill_at_60_resume_from_checkpoint":
                results.append(self._run_kill_resume(scenario, started))
            else:
                results.append(self._run_fault(scenario, started))
        return results

    def _run_fault(self, scenario: Scenario, started: float) -> ScenarioResult:
        victim_id: str | None = None
        attacker_id: str | None = None
        observations: list[Observation] = []
        victim_run: dict[str, Any] | None = None

        try:
            if scenario.needs_victim:
                victim = self.runner.via_git_repo(VICTIM_COMMAND)
                victim_id = victim.instance_id
                if victim_id:
                    self.client.wait_for_instance_state(victim_id, {"running"}, timeout=600)

            attacker = self.runner.via_git_repo(scenario.run_command)
            attacker_id = attacker.instance_id
            if not attacker.submitted or not attacker_id:
                return ScenarioResult(
                    scenario=scenario.name,
                    route=Route.GIT_REPO,
                    instance_id=None,
                    workload_run_id=None,
                    started_at_unix=started,
                    duration_seconds=time.time() - started,
                    observations=[
                        Observation("launch", Verdict.ERROR, f"could not launch: {attacker.detail}")
                    ],
                )

            budget = min(scenario.expected_max_seconds, self.options.hang_budget_seconds)
            run, _ = self.client.wait_for_run_terminal(attacker_id, timeout=budget)

            observations.append(self.observer.run_reaches_terminal_status(run))
            observations.extend(self._scenario_specific(scenario, attacker_id, run))

            if scenario.needs_victim and victim_id:
                victim_run, _ = self.client.wait_for_run_terminal(victim_id, timeout=900)
                observations.append(self.observer.victim_survived(victim_run))
        except PlatformError as exc:
            observations.append(Observation("harness", Verdict.ERROR, f"platform error: {exc}"))
        finally:
            self._cleanup(attacker_id)
            self._cleanup(victim_id)

        return ScenarioResult(
            scenario=scenario.name,
            route=Route.GIT_REPO,
            instance_id=attacker_id,
            workload_run_id=None,
            started_at_unix=started,
            duration_seconds=time.time() - started,
            observations=observations,
            victim_survived=_victim_ok(victim_run) if scenario.needs_victim else None,
        )

    def _scenario_specific(
        self, scenario: Scenario, instance_id: str, run: dict[str, Any] | None
    ) -> list[Observation]:
        obs: list[Observation] = []
        name = scenario.name

        if name == "cuda_oom":
            obs.append(self.observer.exit_code_surfaced(run, expect=137))
            obs.append(
                self.observer.message_matches(
                    "failure_reason_is_legible", run, OOM_PATTERNS, "an out-of-memory condition"
                )
            )
            obs.append(self.observer.instance_not_left_nominally_active(instance_id))
            obs.append(self.observer.billing_stopped(instance_id))
            obs.append(
                Observation(
                    "gpu_reclaimed_for_next_tenant",
                    Verdict.UNOBSERVABLE,
                    "Requires launching a fresh instance on the same physical GPU and "
                    "reading free VRAM at start. The platform exposes no device "
                    "affinity, so this cannot be asserted through the API — check "
                    "manually via the provider console.",
                )
            )

        elif name == "host_ram_exhaustion":
            obs.append(
                self.observer.message_matches(
                    "killed_at_container_limit",
                    run,
                    CGROUP_KILL_PATTERNS,
                    "a container-level kill (SIGKILL/137/cgroup OOM)",
                )
            )
            obs.append(self.observer.metrics_kept_reporting(instance_id, F_RAM_USED, "ram"))
            obs.append(self.observer.instance_not_left_nominally_active(instance_id))

        elif name == "hung_network_read":
            obs.append(self._hang_verdict(instance_id, run))
            obs.append(
                self.observer.message_matches(
                    "termination_reason_is_legible",
                    run,
                    TIMEOUT_PATTERNS,
                    "a timeout, runtime limit, or quota stop",
                )
            )
            obs.append(self.observer.idle_gpu_detected(instance_id))
            obs.append(
                Observation(
                    "gate_b_kept_metering",
                    Verdict.UNOBSERVABLE,
                    "Gate B's poll cadence is only visible in usage_records and the "
                    "admin surface; /billing/usage would confirm it but is outside "
                    "this harness's user-scoped credentials.",
                )
            )

        elif name == "disk_exhaustion":
            obs.append(self.observer.exit_code_surfaced(run, expect=28))
            obs.append(
                self.observer.message_matches(
                    "enospc_error_is_legible", run, ENOSPC_PATTERNS, "a full disk / ENOSPC"
                )
            )
            obs.append(self.observer.platform_retained_ssh_observability(instance_id))
            obs.append(self.observer.disk_metrics_showed_ramp(instance_id))
            obs.append(self.observer.terminate_still_works(instance_id))
            obs.append(
                Observation(
                    "volume_reclaimed_on_teardown",
                    Verdict.UNOBSERVABLE,
                    "Provider-side storage reclamation is not visible through the "
                    "platform API. Verify against the provider account directly.",
                )
            )
        return obs

    def _hang_verdict(self, instance_id: str, run: dict[str, Any] | None) -> Observation:
        key = "something_eventually_terminates_the_job"
        status = pluck(run, ("status",))
        if status.value and str(status.value).lower() in {"failed", "succeeded"}:
            return Observation(
                key,
                Verdict.PASS,
                f"Run terminated with status {status.value!r} within the budget.",
            )
        state = self.client.instance_status(instance_id)
        current = str(state.value).lower() if state.value else "unknown"
        if current in DEAD_INSTANCE_STATES or current == "grace_stopped":
            return Observation(
                key,
                Verdict.PASS,
                f"Instance reached {current!r}, so a backstop fired even though the "
                "run row did not resolve.",
            )
        return Observation(
            key,
            Verdict.FAIL,
            f"After {self.options.hang_budget_seconds:.0f}s the job is still alive "
            f"(instance {current!r}) with zero GPU work done. Nothing — not "
            "ExpiryWorker, not Gate B — reclaimed it. This bills indefinitely.",
            {"instance_status": current},
        )

    def _run_kill_resume(self, scenario: Scenario, started: float) -> ScenarioResult:
        """Launch with a volume, kill at ~60% of steps, relaunch and resume."""
        from .adversarial.kill_resume import RESUME_COMMAND

        observations: list[Observation] = []
        first_id: str | None = None
        second_id: str | None = None

        try:
            first = self.runner.via_git_repo(
                scenario.run_command,
                new_volume_name=f"conformance-ckpt-{int(started)}",
                new_volume_size_gb=50,
            )
            first_id = first.instance_id
            if not first_id:
                raise PlatformError(f"could not launch with volume: {first.detail}")

            reached = self._wait_for_progress_pct(first_id, 60.0, timeout=3600)
            observations.append(
                Observation(
                    "checkpoint_written_before_kill",
                    Verdict.PASS if reached else Verdict.FAIL,
                    f"Training reached {reached or 0:.0f}% before the kill."
                    if reached
                    else "Never observed a progress line reaching 60%; nothing to resume from.",
                )
            )
            if not reached:
                return self._finish_kill_resume(
                    scenario, started, first_id, second_id, observations
                )

            self.client.terminate(first_id)
            self.client.wait_for_instance_state(first_id, DEAD_INSTANCE_STATES, timeout=600)

            second = self.runner.via_git_repo(RESUME_COMMAND)
            second_id = second.instance_id
            run, _ = self.client.wait_for_run_terminal(second_id or "", timeout=5400)
            manifest = _manifest_from_stdout(run)

            observations.append(
                Observation(
                    "checkpoint_survived_termination",
                    Verdict.PASS if manifest else Verdict.UNOBSERVABLE,
                    "Resumed run produced a manifest."
                    if manifest
                    else "No manifest from the resumed run; cannot confirm the "
                    "checkpoint survived.",
                )
            )
            resumed_step = (manifest or {}).get("resumed_from_step")
            observations.append(
                Observation(
                    "resumed_from_nonzero_step",
                    Verdict.PASS if resumed_step else Verdict.FAIL,
                    f"Resumed at step {resumed_step}."
                    if resumed_step
                    else "Resumed run started from step 0 — it silently retrained from "
                    "scratch rather than resuming. This is the false pass this "
                    "scenario exists to catch.",
                    {"resumed_from_step": resumed_step},
                )
            )
            observations.append(
                Observation(
                    "relaunch_reattached_same_volume",
                    Verdict.PASS if resumed_step else Verdict.FAIL,
                    "Volume data was present on the new instance."
                    if resumed_step
                    else "No checkpoint found on the relaunched instance's volume.",
                )
            )
            observations.append(
                Observation(
                    "resumed_metrics_match_baseline",
                    Verdict.UNOBSERVABLE,
                    "Needs an uninterrupted baseline run of the identical config to "
                    "compare against. Run `conformance baseline` first, then re-run "
                    "this scenario with --baseline-manifest.",
                )
            )
            for key, detail in (
                (
                    "usage_metered_across_both_instances",
                    "Requires /billing/usage, which is outside this harness's scope.",
                ),
                (
                    "lineage_is_reconstructable",
                    "Spec section 18 documents this as a known gap: relaunch composes "
                    "already-audited primitives without tagging the pair as one event.",
                ),
            ):
                observations.append(Observation(key, Verdict.UNOBSERVABLE, detail))

        except PlatformError as exc:
            observations.append(Observation("harness", Verdict.ERROR, f"platform error: {exc}"))
        finally:
            self._cleanup(first_id)
            self._cleanup(second_id)

        return self._finish_kill_resume(scenario, started, first_id, second_id, observations)

    def _finish_kill_resume(
        self,
        scenario: Scenario,
        started: float,
        first_id: str | None,
        second_id: str | None,
        observations: list[Observation],
    ) -> ScenarioResult:
        return ScenarioResult(
            scenario=scenario.name,
            route=Route.GIT_REPO,
            instance_id=second_id or first_id,
            workload_run_id=None,
            started_at_unix=started,
            duration_seconds=time.time() - started,
            observations=observations,
            notes=[f"first_instance={first_id}", f"resumed_instance={second_id}"],
        )

    def _wait_for_progress_pct(
        self, instance_id: str, target_pct: float, timeout: float, poll: float = 20.0
    ) -> float | None:
        """Poll stdout for CONFORMANCE-PROGRESS lines until target_pct is reached.

        Wall-clock fractions would fire at the wrong point on a faster or slower
        GPU, so the kill trigger is tied to actual training progress instead.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = self.client.latest_workload_run(instance_id)
            stdout = str(pluck(run, ("stdout",)).value or "")
            best = 0.0
            for line in stdout.splitlines():
                if PROGRESS_MARKER in line:
                    try:
                        import json as _json

                        payload = _json.loads(line.split(PROGRESS_MARKER, 1)[1].strip())
                        best = max(best, float(payload.get("pct", 0.0)))
                    except (ValueError, KeyError):
                        continue
            if best >= target_pct:
                return best
            time.sleep(poll)
        return None

    # --- helpers -----------------------------------------------------------

    def _cleanup(self, instance_id: str | None) -> None:
        """Never leave a billable instance behind, even on a harness crash."""
        if not instance_id:
            return
        with contextlib.suppress(PlatformError):
            self.client.terminate(instance_id)

    def _dry(self, name: str, route: Route, started: float) -> ScenarioResult:
        return ScenarioResult(
            scenario=name,
            route=route,
            instance_id=None,
            workload_run_id=None,
            started_at_unix=started,
            duration_seconds=0.0,
            observations=[Observation("dry_run", Verdict.SKIPPED, "dry run: nothing launched")],
        )


def _victim_ok(run: dict[str, Any] | None) -> bool | None:
    if run is None:
        return None
    status = pluck(run, ("status",))
    return str(status.value).lower() == "succeeded" if status.value else None


def _manifest_from_stdout(run: dict[str, Any] | None) -> dict[str, Any] | None:
    """Recover the workload manifest that the job printed before dying."""
    import json as _json

    from .workloads.base import MANIFEST_MARKER

    stdout = str(pluck(run, ("stdout",)).value or "")
    if MANIFEST_MARKER not in stdout:
        return None
    # The manifest path is on the sandbox's filesystem, which is gone by now, so
    # recover the metrics from the log lines the workload printed instead.
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                return {"metrics": _json.loads(stripped)}
            except ValueError:
                continue
    return None
