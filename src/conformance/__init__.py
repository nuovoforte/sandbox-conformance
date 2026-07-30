"""Conformance harness for the AI Sandbox Pilot platform.

Validates the platform against real AI workloads under both normal and failure
conditions, across every delivery route it exposes.

Two halves that never share a process:

  IN the sandbox   conformance.workloads.*   — three real workloads
                   conformance.adversarial.* — five deliberate faults
  OUTSIDE it       conformance.observer      — asserts what the PLATFORM did
                   conformance.orchestrator  — runs the matrix
                   conformance.report        — scorecard

The split is the whole design. A job that has been OOM-killed, hung, or had its
filesystem filled cannot report on how the platform handled it, so any harness
where the failing job grades itself is measuring nothing.
"""

__version__ = "0.1.0"
