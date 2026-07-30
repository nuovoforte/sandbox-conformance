"""The shared contract every workload, scenario, and route agrees on.

One schema for results across three real workloads, five adversarial scenarios,
and five delivery routes. Without this, "did the platform behave correctly" has a
different answer shape per test and nothing can be aggregated into a scorecard.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Verdict(StrEnum):
    """Outcome of one expectation.

    UNOBSERVABLE is not a soft FAIL — it is a distinct and often more serious
    finding. It means the platform may have behaved correctly but exposes no way
    for anyone outside the sandbox to know. A platform that cannot be observed
    cannot be operated.
    """

    PASS = "pass"
    FAIL = "fail"
    UNOBSERVABLE = "unobservable"
    SKIPPED = "skipped"
    ERROR = "error"


class Route(StrEnum):
    """Delivery routes the platform exposes, per Master Spec sections 7/19/25."""

    CLI = "cli"
    SDK = "sdk"
    REST_API = "rest_api"
    GIT_REPO = "git_repo"
    NOTEBOOK = "notebook"
    SSH = "ssh"


@dataclass(frozen=True, slots=True)
class Expectation:
    """One assertion about platform behaviour, checked from outside the sandbox.

    `key` is stable across runs so a scorecard can diff two runs. `spec_ref`
    points at the Master Spec clause the expectation enforces, so a FAIL is
    arguable against the document rather than against my opinion.
    """

    key: str
    describes: str
    spec_ref: str
    blast_radius: bool = False


@dataclass(slots=True)
class Observation:
    """The result of checking one Expectation."""

    key: str
    verdict: Verdict
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["verdict"] = self.verdict.value
        return payload


@dataclass(slots=True)
class ScenarioResult:
    """Everything one scenario run produced, normal or adversarial."""

    scenario: str
    route: Route
    instance_id: str | None
    workload_run_id: str | None
    started_at_unix: float
    duration_seconds: float
    observations: list[Observation] = field(default_factory=list)
    workload_metrics: dict[str, Any] | None = None
    victim_survived: bool | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> Verdict:
        """Worst-case rollup. ERROR > FAIL > UNOBSERVABLE > PASS."""
        order = [
            Verdict.ERROR,
            Verdict.FAIL,
            Verdict.UNOBSERVABLE,
            Verdict.PASS,
            Verdict.SKIPPED,
        ]
        seen = {o.verdict for o in self.observations}
        for verdict in order:
            if verdict in seen:
                return verdict
        return Verdict.SKIPPED

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "route": self.route.value,
            "verdict": self.verdict.value,
            "instance_id": self.instance_id,
            "workload_run_id": self.workload_run_id,
            "started_at_unix": round(self.started_at_unix, 3),
            "duration_seconds": round(self.duration_seconds, 2),
            "victim_survived": self.victim_survived,
            "observations": [o.to_dict() for o in self.observations],
            "workload_metrics": self.workload_metrics,
            "notes": self.notes,
        }


@dataclass(slots=True)
class ConformanceReport:
    """The full matrix run."""

    results: list[ScenarioResult] = field(default_factory=list)
    platform_base_url: str = ""
    started_at_unix: float = 0.0
    duration_seconds: float = 0.0
    skipped: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {v.value: 0 for v in Verdict}
        for result in self.results:
            tally[result.verdict.value] += 1
        return tally

    def failures(self) -> list[ScenarioResult]:
        return [r for r in self.results if r.verdict in (Verdict.FAIL, Verdict.ERROR)]

    def unobservable(self) -> list[ScenarioResult]:
        return [r for r in self.results if r.verdict is Verdict.UNOBSERVABLE]

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform_base_url": self.platform_base_url,
            "started_at_unix": round(self.started_at_unix, 3),
            "duration_seconds": round(self.duration_seconds, 2),
            "counts": self.counts(),
            "skipped": self.skipped,
            "results": [r.to_dict() for r in self.results],
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path
