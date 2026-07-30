"""An in-memory stand-in for the platform API.

Lets the observer's verdict logic be tested against every interesting shape:
correct behaviour, incorrect behaviour, and — the case that matters most — a
platform that behaves correctly but exposes no field to prove it.
"""

from __future__ import annotations

from typing import Any


class FakePlatformClient:
    """Duck-types the parts of PlatformClient the Observer touches."""

    base_url = "http://fake.local"

    def __init__(
        self,
        *,
        instance_status: str | None = "terminated",
        runs: list[dict[str, Any]] | None = None,
        metrics: list[dict[str, Any]] | None = None,
        latest_metrics: dict[str, Any] | None = None,
        terminate_raises: Exception | None = None,
        status_raises: Exception | None = None,
    ) -> None:
        self._status = instance_status
        self._runs = runs if runs is not None else []
        self._metrics = metrics or []
        self._latest = latest_metrics
        self._terminate_raises = terminate_raises
        self._status_raises = status_raises
        self.terminated: list[str] = []

    # --- surface used by Observer -----------------------------------------

    def instance(self, instance_id: str) -> dict[str, Any]:
        if self._status_raises:
            raise self._status_raises
        return {"id": instance_id, "status": self._status} if self._status else {"id": instance_id}

    def instance_status(self, instance_id: str) -> Any:
        from conformance.platform_api import pluck

        return pluck(self.instance(instance_id), ("status", "instance_status"))

    def workload_runs(self, instance_id: str) -> list[dict[str, Any]]:
        return self._runs

    def latest_workload_run(self, instance_id: str) -> dict[str, Any] | None:
        return self._runs[0] if self._runs else None

    def metrics_series(self, instance_id: str, since: str | None = None) -> list[dict[str, Any]]:
        return self._metrics

    def metrics_latest(self, instance_id: str) -> dict[str, Any] | None:
        return self._latest

    def terminate(self, instance_id: str) -> None:
        if self._terminate_raises:
            raise self._terminate_raises
        self.terminated.append(instance_id)
        self._status = "terminated"

    def wait_for_instance_state(
        self, instance_id: str, states: set[str], timeout: float, poll: float = 10.0
    ) -> tuple[str | None, float]:
        del states, timeout, poll
        return self._status, 0.0

    def wait_for_run_terminal(
        self, instance_id: str, timeout: float, poll: float = 10.0
    ) -> tuple[dict[str, Any] | None, float]:
        return (self._runs[0] if self._runs else None), 0.0
