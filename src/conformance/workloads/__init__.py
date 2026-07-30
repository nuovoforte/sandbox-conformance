"""The three real workloads, behind one uniform contract.

Each exposes `run` with a `--canary` flag and emits the same manifest schema, so
the route matrix and the adversarial orchestrator can treat them interchangeably.
"""

from __future__ import annotations

WORKLOAD_MODULES = {
    "lora": "conformance.workloads.lora",
    "whisper": "conformance.workloads.whisper",
    "faiss": "conformance.workloads.embed_faiss",
}


# Command each route uses to invoke a workload inside the sandbox.
def run_command(workload: str, *, canary: bool = False, extra: str = "") -> str:
    module = WORKLOAD_MODULES[workload]
    flag = " --canary" if canary else ""
    return f"python -m {module} run{flag} {extra}".strip()


__all__ = ["WORKLOAD_MODULES", "run_command"]
