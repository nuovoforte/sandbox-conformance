"""Scenario definition shared by all five adversarial cases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..contract import Expectation


@dataclass(frozen=True, slots=True)
class Scenario:
    """One adversarial case: how to break it, and what the platform owes us."""

    name: str
    summary: str
    run_command: str
    """Shell command the platform executes inside the sandbox to cause the fault."""

    expectations: tuple[Expectation, ...]

    needs_victim: bool = False
    """True when the fault is node-level, so blast radius is the real assertion.

    A victim job runs concurrently on the same node; if it dies alongside the
    attacker, one tenant's mistake took out another's work, which at 50 real
    developers is the difference between a usable platform and an incident.
    """

    needs_gpu: bool = True
    expected_max_seconds: float = 1800.0
    """How long to wait for the platform to act before calling it a hang."""

    requires_volume: bool = False
    cause: Callable[..., int] | None = field(default=None, compare=False)
    """In-sandbox entrypoint, for running a scenario directly without the platform."""

    def expectation(self, key: str) -> Expectation:
        for exp in self.expectations:
            if exp.key == key:
                return exp
        raise KeyError(f"{self.name} declares no expectation {key!r}")
