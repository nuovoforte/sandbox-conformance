"""Delivery routes — the "how does the workload get to the platform" axis.

Every route launches the SAME workload and is judged on the same four questions:
can it submit, can it stream progress, can it retrieve the artifact, can it report
exit status. That uniformity is what makes route results comparable; without it,
"Jupyter works" and "the CLI works" would mean different things.

A correction to what you told me, based on Master Spec v27:

  You listed "Jupyter notebook server" as a working route. The spec describes
  something materially different. Section 25 defines the notebook route as an
  uploaded `.ipynb` executed **headlessly with papermill** by `WorkloadRunWorker`,
  with the output notebook downloadable afterwards. There is a `Jupyter` entry in
  the framework template catalog (section 6), so a container can *serve* Jupyter
  and a user can reach it through `connection_details` — but that is an
  unmanaged, unobserved surface: no `workload_runs` row, no exit code, no
  `WORKLOAD_RUN_SUCCEEDED` audit event.

  These are two different routes with two different risk profiles, so they are
  modelled separately below. Conflating them would mean testing papermill and
  reporting it as interactive Jupyter, which is exactly the kind of false
  assurance this harness exists to prevent.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from .contract import Observation, Route, Verdict
from .platform_api import (
    F_EXIT_CODE,
    F_RUN_STATUS,
    PlatformClient,
    PlatformError,
    pluck,
)
from .workloads.base import MANIFEST_MARKER

DEFAULT_TEMPLATE = "pytorch"


@dataclass(slots=True)
class RouteLaunch:
    """What a route returns after submitting work."""

    route: Route
    instance_id: str | None
    workload_run_id: str | None
    submitted: bool
    detail: str
    stdout: str = ""


class RouteRunner:
    """Launches a workload through one delivery route."""

    def __init__(
        self,
        client: PlatformClient,
        repo_url: str,
        repo_ref: str = "main",
        template: str = DEFAULT_TEMPLATE,
        gpu: str | None = None,
    ) -> None:
        self.client = client
        self.repo_url = repo_url
        self.repo_ref = repo_ref
        self.template = template
        self.gpu = gpu

    # --- routes ------------------------------------------------------------

    def via_git_repo(self, run_command: str, **launch_kwargs: Any) -> RouteLaunch:
        """Spec section 25: launch with workload_source=git_repo."""
        try:
            payload = self.client.launch(
                template_id=self.template,
                gpu=self.gpu,
                workload_source="git_repo",
                git_repo_url=self.repo_url,
                git_repo_ref=self.repo_ref,
                run_command=run_command,
                **launch_kwargs,
            )
        except PlatformError as exc:
            return RouteLaunch(Route.GIT_REPO, None, None, False, f"launch rejected: {exc}")
        return RouteLaunch(
            Route.GIT_REPO,
            _instance_id_of(payload),
            None,
            True,
            "launched with git_repo workload source",
        )

    def via_rest_api(self, run_command: str, **launch_kwargs: Any) -> RouteLaunch:
        """Launch bare, then attach a workload to the running instance (spec 28)."""
        try:
            payload = self.client.launch(template_id=self.template, gpu=self.gpu, **launch_kwargs)
        except PlatformError as exc:
            return RouteLaunch(Route.REST_API, None, None, False, f"launch rejected: {exc}")
        instance_id = _instance_id_of(payload)
        if not instance_id:
            return RouteLaunch(
                Route.REST_API, None, None, False, f"no instance id in launch response: {payload!r}"
            )
        state, _ = self.client.wait_for_instance_state(instance_id, {"running"}, timeout=600)
        if state != "running":
            return RouteLaunch(
                Route.REST_API,
                instance_id,
                None,
                False,
                f"instance never reached running ({state})",
            )
        try:
            # workload_source=git_repo requires the repo URL too — attaching a
            # bare command is not a shape the endpoint accepts (422, found on
            # the first live pass). The canary clones this repo like every
            # other git_repo submission.
            run = self.client.attach_workload(
                instance_id, git_repo_url=self.repo_url, run_command=run_command
            )
        except PlatformError as exc:
            return RouteLaunch(
                Route.REST_API, instance_id, None, False, f"attach-workload rejected: {exc}"
            )
        return RouteLaunch(
            Route.REST_API, instance_id, _run_id_of(run), True, "attached via POST workload-run"
        )

    def via_notebook(self, workload: str, canary: bool) -> RouteLaunch:
        """Spec section 25: upload a .ipynb, executed headlessly by papermill.

        This is NOT an interactive Jupyter server — see this module's docstring.
        """
        notebook = build_notebook(workload, canary=canary)
        try:
            uploaded = self.client.upload_notebook(f"{workload}_conformance.ipynb", notebook)
        except PlatformError as exc:
            return RouteLaunch(Route.NOTEBOOK, None, None, False, f"notebook upload failed: {exc}")
        upload_id = pluck(uploaded, ("id", "notebook_upload_id")).value
        if not upload_id:
            return RouteLaunch(
                Route.NOTEBOOK, None, None, False, f"no upload id in response: {uploaded!r}"
            )
        try:
            payload = self.client.launch(
                template_id=self.template,
                gpu=self.gpu,
                workload_source="notebook",
                notebook_upload_id=upload_id,
            )
        except PlatformError as exc:
            return RouteLaunch(Route.NOTEBOOK, None, None, False, f"launch rejected: {exc}")
        return RouteLaunch(
            Route.NOTEBOOK, _instance_id_of(payload), None, True, "launched via papermill notebook"
        )

    def via_cli(self, run_command: str, cli_binary: str = "ai-sandbox") -> RouteLaunch:
        """Spec section 7: `ai-sandbox launch --template X --git-repo ... --run-command ...`.

        Shells out to the real CLI rather than reimplementing it, because the CLI
        is the thing under test — including its argument parsing and its claim to
        stream progress and print a connection URL to stdout.
        """
        argv = [
            cli_binary,
            "launch",
            "--template",
            self.template,
            "--git-repo",
            self.repo_url,
            "--run-command",
            run_command,
        ]
        if self.gpu:
            argv += ["--gpu", self.gpu]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=900, check=False)
        except FileNotFoundError:
            return RouteLaunch(
                Route.CLI,
                None,
                None,
                False,
                f"{cli_binary!r} not on PATH; install the platform CLI",
            )
        except subprocess.TimeoutExpired:
            return RouteLaunch(
                Route.CLI, None, None, False, "CLI launch did not return within 900s"
            )
        if proc.returncode != 0:
            return RouteLaunch(
                Route.CLI, None, None, False, f"CLI exited {proc.returncode}: {proc.stderr[-400:]}"
            )
        return RouteLaunch(
            Route.CLI,
            _scrape_instance_id(proc.stdout),
            None,
            True,
            "launched via CLI",
            stdout=proc.stdout,
        )

    def via_sdk(self, run_command: str) -> RouteLaunch:
        """Spec section 7: `Client().launch(...)` from the platform's Python SDK."""
        try:
            import ai_sandbox  # type: ignore[import-not-found]
        except ImportError:
            return RouteLaunch(
                Route.SDK, None, None, False, "platform SDK (`ai_sandbox`) not importable"
            )
        try:
            client = ai_sandbox.Client()
            result = client.launch(
                template=self.template,
                workload_source="git_repo",
                git_repo_url=self.repo_url,
                run_command=run_command,
            )
        except Exception as exc:
            return RouteLaunch(Route.SDK, None, None, False, f"SDK launch raised {exc!r}")
        instance_id = getattr(result, "id", None) or _instance_id_of(
            result if isinstance(result, dict) else {}
        )
        return RouteLaunch(Route.SDK, instance_id, None, True, "launched via SDK")


# --- route conformance checks ----------------------------------------------


def check_route(
    client: PlatformClient,
    launch: RouteLaunch,
    timeout: float = 1800.0,
) -> list[Observation]:
    """The four questions asked of every route, identically."""
    observations: list[Observation] = []
    prefix = launch.route.value

    observations.append(
        Observation(
            f"{prefix}.can_submit",
            Verdict.PASS if launch.submitted else Verdict.FAIL,
            launch.detail,
        )
    )
    if not launch.submitted or not launch.instance_id:
        for suffix in ("can_report_status", "can_retrieve_artifact", "can_report_exit_code"):
            observations.append(
                Observation(f"{prefix}.{suffix}", Verdict.SKIPPED, "submission did not succeed")
            )
        return observations

    run, waited = client.wait_for_run_terminal(launch.instance_id, timeout=timeout)

    status = pluck(run, F_RUN_STATUS)
    if not status.observable:
        observations.append(
            Observation(
                f"{prefix}.can_report_status",
                Verdict.UNOBSERVABLE,
                f"No run status retrievable after {waited:.0f}s via this route.",
            )
        )
    else:
        value = str(status.value).lower()
        observations.append(
            Observation(
                f"{prefix}.can_report_status",
                Verdict.PASS if value in {"succeeded", "failed"} else Verdict.FAIL,
                f"Run status {value!r} after {waited:.0f}s.",
                {"status": value},
            )
        )

    stdout = str(pluck(run, ("stdout",)).value or "") + launch.stdout
    if MANIFEST_MARKER in stdout:
        path = stdout.split(MANIFEST_MARKER, 1)[1].strip().splitlines()[0]
        observations.append(
            Observation(
                f"{prefix}.can_retrieve_artifact",
                Verdict.PASS,
                f"Workload manifest located at {path}.",
                {"manifest_path": path},
            )
        )
    else:
        observations.append(
            Observation(
                f"{prefix}.can_retrieve_artifact",
                Verdict.FAIL if stdout else Verdict.UNOBSERVABLE,
                "No manifest marker found in captured output; results cannot be "
                "retrieved through this route.",
            )
        )

    code = pluck(run, F_EXIT_CODE)
    observations.append(
        Observation(
            f"{prefix}.can_report_exit_code",
            Verdict.PASS if code.observable and code.value is not None else Verdict.UNOBSERVABLE,
            f"exit_code={code.value!r} (field {code.found_as!r})",
        )
    )
    return observations


# --- helpers ----------------------------------------------------------------


def _instance_id_of(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("instance_id", "sandbox_instance_id", "id"):
        if payload.get(key):
            return str(payload[key])
    nested = payload.get("instance")
    if isinstance(nested, dict):
        return _instance_id_of(nested)
    return None


def _run_id_of(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("workload_run_id", "id"):
        if payload.get(key):
            return str(payload[key])
    return None


def _scrape_instance_id(stdout: str) -> str | None:
    """Pull a UUID-ish token out of CLI output."""
    import re

    match = re.search(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", stdout, re.IGNORECASE
    )
    return match.group(0) if match else None


def build_notebook(workload: str, canary: bool = False) -> bytes:
    """Generate a minimal .ipynb that runs a workload under papermill.

    Written as a real notebook rather than a template file on disk so the
    notebook route has no extra deployment step, and so the command inside it
    always matches the workloads package in this repo.
    """
    from .workloads import run_command

    command = run_command(workload, canary=canary)
    source = [
        "import subprocess, sys\n",
        f"cmd = {shlex.split(command)!r}\n",
        "proc = subprocess.run(cmd, capture_output=True, text=True)\n",
        "print(proc.stdout)\n",
        "print(proc.stderr, file=sys.stderr)\n",
        # Non-zero exit must fail the notebook, or papermill reports success for
        # a workload that crashed and the route check becomes meaningless.
        "assert proc.returncode == 0, f'workload exited {proc.returncode}'\n",
    ]
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"tags": ["conformance"]},
                "outputs": [],
                "source": source,
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1).encode()


def probe_interactive_jupyter(client: PlatformClient, instance_id: str) -> Observation:
    """Is there an interactive Jupyter server, distinct from papermill execution?

    Reported separately and honestly. If a Jupyter URL is reachable, note that
    anything a developer runs in it produces no workload_runs row, no exit code,
    and no audit event — real usage the platform cannot see, bill against, or
    debug.
    """
    key = "interactive_jupyter_is_observable"
    try:
        details = client.instance(instance_id)
    except PlatformError as exc:
        return Observation(key, Verdict.ERROR, f"could not read instance: {exc}")
    blob = json.dumps(details).lower() if details else ""
    if "jupyter" not in blob and "8888" not in blob:
        return Observation(
            key,
            Verdict.SKIPPED,
            "No Jupyter endpoint advertised in connection details for this template.",
        )
    started = time.time()
    runs_before = len(client.workload_runs(instance_id))
    return Observation(
        key,
        Verdict.UNOBSERVABLE,
        "A Jupyter endpoint is advertised, but work done in an interactive session "
        "creates no workload_runs row, no exit code, and no audit event. Anything a "
        "developer runs there is invisible to the platform — untracked, undebuggable, "
        "and outside the failure-detection path spec section 6 promises.",
        {"runs_visible": runs_before, "checked_at": round(started, 3)},
    )
