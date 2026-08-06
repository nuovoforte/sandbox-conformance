"""Client for the AI Sandbox platform's own REST surface.

EVERY endpoint path and response field name the harness depends on is defined
here and nowhere else. The Master Spec (v27) documents 87 endpoints and the
column names behind them, but not the response body shapes, so some field names
below are inferred from table columns. When one is wrong, the fix is a single
constant in this file rather than a hunt through the observer.

Field access goes through `_pluck`, which tries several candidate names and
reports which one it found. A field that resolves to nothing becomes an
UNOBSERVABLE verdict upstream rather than a crash or a false FAIL.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

# --- Endpoint paths (Master Spec section 19) --------------------------------

EP_LOGIN = "/auth/login"
EP_REFRESH = "/auth/refresh"
EP_TEMPLATES = "/templates"
EP_LAUNCH = "/sandbox/launch"
EP_INSTANCES = "/sandbox/instances"
EP_INSTANCE = "/sandbox/instances/{id}"
EP_STOP = "/sandbox/instances/{id}/stop"
EP_TERMINATE = "/sandbox/instances/{id}/terminate"
EP_METRICS_LATEST = "/sandbox/instances/{id}/metrics/latest"
EP_METRICS_SERIES = "/sandbox/instances/{id}/metrics"
EP_WORKLOAD_RUNS = "/sandbox/instances/{id}/workload-runs"
EP_WORKLOAD_RUN = "/sandbox/instances/{id}/workload-run"
EP_WORKLOAD_RERUN = "/sandbox/instances/{id}/workload-run/rerun"
EP_NOTEBOOK_UPLOAD = "/workloads/notebooks"
EP_FILES = "/sandbox/instances/{id}/files"
EP_AUDIT_ME = "/audit-logs/me"
EP_QUOTA_POOL = "/quota/multi-gpu-pool"
EP_PROVIDER_HEALTH = "/providers/health"
EP_HEALTHZ = "/healthz"
EP_VOLUMES = "/volumes"

# --- Field-name candidates (inferred from Master Spec section 17) -----------

F_INSTANCE_ID = ("id", "instance_id", "sandbox_instance_id")
F_INSTANCE_STATUS = ("status", "instance_status")
F_RUN_ID = ("id", "workload_run_id")
F_RUN_STATUS = ("status", "workload_run_status")
F_EXIT_CODE = ("exit_code", "exitcode")
F_FAILURE_REASON = ("failure_reason", "failureReason", "error")
F_STDOUT = ("stdout", "std_out")
F_STDERR = ("stderr", "std_err")
F_DISK_USED = ("disk_used_gb", "diskUsedGb")
F_DISK_TOTAL = ("disk_total_gb", "diskTotalGb")
F_RAM_USED = ("ram_used_mb", "ramUsedMb")
F_GPU_UTIL = ("gpu_utilization_pct_avg", "gpuUtilizationPctAvg")
F_COLLECTED_AT = ("collected_at", "collectedAt", "created_at")

# Terminal instance states (Master Spec section 18, InstanceStatus)
TERMINAL_INSTANCE_STATES = {
    "runtime_failed",
    "stopped",
    "grace_stopped",
    "terminated",
    "hard_terminated",
}
DEAD_INSTANCE_STATES = {"terminated", "hard_terminated"}
TERMINAL_RUN_STATES = {"succeeded", "failed"}

MISSING = object()


class PlatformError(RuntimeError):
    """The platform API refused or failed a request."""


@dataclass(slots=True)
class Plucked:
    """A field lookup result that records whether and how it was found."""

    value: Any
    found_as: str | None

    @property
    def observable(self) -> bool:
        return self.found_as is not None


def pluck(payload: Any, candidates: tuple[str, ...]) -> Plucked:
    """Read the first present candidate key from a dict-ish payload."""
    if not isinstance(payload, dict):
        return Plucked(None, None)
    for name in candidates:
        if name in payload and payload[name] is not None:
            return Plucked(payload[name], name)
    # Distinguish "key absent" from "key present but null": a null exit_code on a
    # finished run is a real finding, not an unobservable field.
    for name in candidates:
        if name in payload:
            return Plucked(None, name)
    return Plucked(None, None)


class PlatformClient:
    """Thin, synchronous client. Deliberately not the project's own SDK.

    Using the platform's own Python SDK here would let an SDK bug mask a backend
    bug — the harness must be able to disagree with the SDK. The SDK is exercised
    separately as one of the delivery routes under test.
    """

    def __init__(
        self,
        base_url: str,
        email: str | None = None,
        password: str | None = None,
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._token = token
        # Kept so an expired access token mid-run can be replaced. The first
        # live pass (2026-07-30) ran ~55 minutes, the platform's access-token
        # TTL is shorter, and every scenario after expiry died with 401 —
        # including the orchestrator's own cleanup, which leaked two billable
        # pods. A conformance run must outlive one token.
        self._email = email
        self._password = password
        if token is None and email and password:
            self.login(email, password)

    # --- plumbing ----------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PlatformClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = self._client.request(method, path, headers=self._headers(), **kwargs)
        except httpx.HTTPError as exc:
            raise PlatformError(f"{method} {path} transport failure: {exc}") from exc
        if resp.status_code == 401 and self._email and self._password and path != EP_LOGIN:
            # Access token expired mid-run — re-login once and retry. Without
            # this, a matrix longer than one token TTL loses every remaining
            # scenario AND the cleanup pass (observed live 2026-07-30).
            self.login(self._email, self._password)
            try:
                resp = self._client.request(method, path, headers=self._headers(), **kwargs)
            except httpx.HTTPError as exc:
                raise PlatformError(f"{method} {path} transport failure: {exc}") from exc
        if resp.status_code >= 400:
            raise PlatformError(f"{method} {path} -> {resp.status_code}: {resp.text[:400]}")
        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    # --- auth --------------------------------------------------------------

    def login(self, email: str, password: str) -> None:
        payload = self.request("POST", EP_LOGIN, json={"email": email, "password": password})
        token = pluck(payload, ("access_token", "token", "accessToken"))
        if not token.observable or not token.value:
            raise PlatformError(f"login succeeded but no access token in response: {payload!r}")
        self._token = str(token.value)

    # --- lifecycle ---------------------------------------------------------

    def healthz(self) -> Any:
        # /healthz deliberately stays bare (OI-51) while every other route
        # moved under /v1 -- self.base_url already carries that /v1 suffix
        # (needed for every other endpoint below), so a plain
        # self.request("GET", EP_HEALTHZ) 404s against .../v1/healthz.
        # Hits the origin directly instead, bypassing self._client's
        # /v1-prefixed base_url for this one endpoint only.
        origin = httpx.URL(self.base_url).copy_with(path="", query=None, fragment=None)
        resp = httpx.get(f"{origin}{EP_HEALTHZ}", timeout=self._client.timeout)
        if resp.status_code >= 400:
            raise PlatformError(f"GET {EP_HEALTHZ} -> {resp.status_code}: {resp.text[:400]}")
        return resp.json() if resp.content else None

    def templates(self) -> Any:
        return self.request("GET", EP_TEMPLATES)

    def launch(self, **body: Any) -> Any:
        """POST /sandbox/launch. Only the template is required (spec section 6).

        NOTE: the wire field is `template_id`, not `template`. Spec sections 1
        and 6 both call it "template", and the platform's own Python SDK
        exposes it as `template=` while mapping it to `template_id` in the
        request body -- so the prose is about the SDK's ergonomics, not the
        REST contract. Callers hitting this endpoint directly must send
        `template_id`; the backend rejects anything else with "Provide exactly
        one of template_id or free_text_request."
        """
        return self.request(
            "POST", EP_LAUNCH, json={k: v for k, v in body.items() if v is not None}
        )

    def instance(self, instance_id: str) -> Any:
        return self.request("GET", EP_INSTANCE.format(id=instance_id))

    def instance_status(self, instance_id: str) -> Plucked:
        return pluck(self.instance(instance_id), F_INSTANCE_STATUS)

    def terminate(self, instance_id: str) -> Any:
        return self.request("POST", EP_TERMINATE.format(id=instance_id))

    def stop(self, instance_id: str) -> Any:
        return self.request("POST", EP_STOP.format(id=instance_id))

    # --- workload runs -----------------------------------------------------

    def workload_runs(self, instance_id: str) -> list[dict[str, Any]]:
        payload = self.request("GET", EP_WORKLOAD_RUNS.format(id=instance_id))
        if isinstance(payload, dict):
            for key in ("items", "results", "workload_runs", "data"):
                if isinstance(payload.get(key), list):
                    return payload[key]
            return []
        return payload if isinstance(payload, list) else []

    def latest_workload_run(self, instance_id: str) -> dict[str, Any] | None:
        try:
            payload = self.request("GET", EP_WORKLOAD_RUN.format(id=instance_id))
        except PlatformError:
            runs = self.workload_runs(instance_id)
            return runs[0] if runs else None
        return payload if isinstance(payload, dict) else None

    def attach_workload(
        self,
        instance_id: str,
        *,
        git_repo_url: str | None = None,
        git_repo_ref: str | None = None,
        run_command: str | None = None,
        notebook_upload_id: str | None = None,
    ) -> Any:
        """Attach a first workload to an already-running instance (spec 28).

        `workload_source` is required by the endpoint (422 without it — found
        on the first live pass); derive it from which arguments were given
        rather than asking every caller to know the wire contract."""
        source = "notebook" if notebook_upload_id else "git_repo"
        body = {
            "workload_source": source,
            "git_repo_url": git_repo_url,
            "git_repo_ref": git_repo_ref,
            "run_command": run_command,
            "notebook_upload_id": notebook_upload_id,
        }
        return self.request(
            "POST",
            EP_WORKLOAD_RUN.format(id=instance_id),
            json={k: v for k, v in body.items() if v is not None},
        )

    def rerun_workload(self, instance_id: str, **overrides: Any) -> Any:
        return self.request(
            "POST",
            EP_WORKLOAD_RERUN.format(id=instance_id),
            json={k: v for k, v in overrides.items() if v is not None},
        )

    def upload_notebook(self, filename: str, content: bytes) -> Any:
        return self.request(
            "POST",
            EP_NOTEBOOK_UPLOAD,
            files={"file": (filename, content, "application/x-ipynb+json")},
        )

    def upload_file(self, instance_id: str, filename: str, content: bytes, target_path: str) -> Any:
        return self.request(
            "POST",
            EP_FILES.format(id=instance_id),
            files={"file": (filename, content, "application/octet-stream")},
            data={"target_path": target_path},
        )

    # --- observability -----------------------------------------------------

    def metrics_latest(self, instance_id: str) -> dict[str, Any] | None:
        try:
            payload = self.request("GET", EP_METRICS_LATEST.format(id=instance_id))
        except PlatformError:
            return None
        return payload if isinstance(payload, dict) else None

    def metrics_series(self, instance_id: str, since: str | None = None) -> list[dict[str, Any]]:
        params = {"since": since} if since else None
        try:
            payload = self.request("GET", EP_METRICS_SERIES.format(id=instance_id), params=params)
        except PlatformError:
            return []
        if isinstance(payload, dict):
            for key in ("items", "results", "metrics", "data"):
                if isinstance(payload.get(key), list):
                    return payload[key]
            return []
        return payload if isinstance(payload, list) else []

    def audit_log_me(self) -> list[dict[str, Any]]:
        try:
            payload = self.request("GET", EP_AUDIT_ME)
        except PlatformError:
            return []
        if isinstance(payload, dict):
            for key in ("items", "results", "logs", "data"):
                if isinstance(payload.get(key), list):
                    return payload[key]
            return []
        return payload if isinstance(payload, list) else []

    # --- polling helpers ---------------------------------------------------

    def wait_for_instance_state(
        self,
        instance_id: str,
        states: set[str],
        timeout: float,
        poll: float = 10.0,
    ) -> tuple[str | None, float]:
        """Poll until the instance reaches one of `states`. Returns (state, waited)."""
        deadline = time.monotonic() + timeout
        started = time.monotonic()
        last: str | None = None
        while time.monotonic() < deadline:
            found = self.instance_status(instance_id)
            last = str(found.value).lower() if found.value is not None else None
            if last in states:
                return last, time.monotonic() - started
            time.sleep(poll)
        return last, time.monotonic() - started

    def wait_for_run_terminal(
        self,
        instance_id: str,
        timeout: float,
        poll: float = 10.0,
    ) -> tuple[dict[str, Any] | None, float]:
        """Poll the latest workload run until it reaches a terminal status."""
        deadline = time.monotonic() + timeout
        started = time.monotonic()
        run: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            run = self.latest_workload_run(instance_id)
            status = pluck(run, F_RUN_STATUS)
            if status.value and str(status.value).lower() in TERMINAL_RUN_STATES:
                return run, time.monotonic() - started
            time.sleep(poll)
        return run, time.monotonic() - started
