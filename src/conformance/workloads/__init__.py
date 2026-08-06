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


# Command each route uses to invoke a workload inside the sandbox. The
# cloned checkout at the sandbox's cwd IS this repo, but `conformance` is
# never installed into it — every prior git_repo-routed workload run 404'd
# with "ModuleNotFoundError: No module named 'conformance'" before a single
# line of workload code ran (surfaced live 2026-08-06: this is also why the
# host_ram_exhaustion/disk_exhaustion adversarial scenarios' "victim" job
# always died, misread as a cross-tenant isolation failure when it was this
# bug). pyproject.toml's own `workloads` extra says as much ("Installed
# INSIDE the sandbox, by the workload itself") — this just never happened.
def run_command(workload: str, *, canary: bool = False, extra: str = "") -> str:
    module = WORKLOAD_MODULES[workload]
    flag = " --canary" if canary else ""
    install = 'pip install -q ".[workloads]"'
    return f"{install} && python -m {module} run{flag} {extra}".strip()


__all__ = ["WORKLOAD_MODULES", "run_command"]
