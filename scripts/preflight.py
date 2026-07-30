#!/usr/bin/env python3
"""Report what the sandbox actually gave us, before any real work starts.

Run this first on any new sandbox. It answers, in one screen: which GPU, how
much VRAM, which driver and CUDA runtime, how much disk and RAM, whether egress
to the model registry works, and whether the output directory is writable.

Deliberately depends only on the standard library plus an optional torch probe,
so it still produces a useful report on a sandbox where the main dependency
install failed.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

EGRESS_TARGETS = (
    "https://huggingface.co",
    "https://pypi.org",
)


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def probe_host() -> dict[str, Any]:
    total, used, free = shutil.disk_usage("/")
    gib = 1024**3
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "disk_total_gib": round(total / gib, 1),
        "disk_free_gib": round(free / gib, 1),
        "disk_used_pct": round(100 * used / total, 1) if total else None,
    }


def probe_nvidia_smi() -> dict[str, Any]:
    query = "name,memory.total,memory.used,driver_version,compute_cap"
    raw = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"])
    if raw is None:
        return {"available": False, "reason": "nvidia-smi missing or returned non-zero"}

    gpus = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        gpus.append(
            {
                "name": parts[0],
                "memory_total": parts[1],
                "memory_used": parts[2],
                "driver_version": parts[3],
                "compute_capability": parts[4],
            }
        )
    return {"available": True, "count": len(gpus), "gpus": gpus}


def probe_torch() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False, "reason": "torch not importable"}

    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    if not torch.cuda.is_available():
        return info

    info["device_count"] = torch.cuda.device_count()
    info["devices"] = [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "vram_gib": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
        }
        for i in range(torch.cuda.device_count())
    ]

    # Smallest possible proof that a kernel actually launches. A sandbox can
    # report a GPU and still fail here on driver/runtime mismatch.
    try:
        start = time.perf_counter()
        a = torch.randn(1024, 1024, device="cuda")
        result = float((a @ a).sum().item())
        torch.cuda.synchronize()
        info["matmul_smoke_test"] = {
            "ok": True,
            "seconds": round(time.perf_counter() - start, 4),
            "finite": result == result,  # NaN check without importing math
        }
    except Exception as exc:
        info["matmul_smoke_test"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return info


def probe_egress(targets: tuple[str, ...] = EGRESS_TARGETS) -> dict[str, Any]:
    results = {}
    for url in targets:
        start = time.perf_counter()
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=8) as resp:
                results[url] = {
                    "ok": True,
                    "status": resp.status,
                    "seconds": round(time.perf_counter() - start, 3),
                }
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            results[url] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return results


def probe_writable(path: str = "artifacts") -> dict[str, Any]:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_probe")
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
        return {"ok": True, "path": os.path.abspath(path)}
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    report = {
        "host": probe_host(),
        "nvidia_smi": probe_nvidia_smi(),
        "torch": probe_torch(),
        "egress": probe_egress(),
        "artifact_dir_writable": probe_writable(),
    }
    print(json.dumps(report, indent=2))

    gpu_ok = report["torch"].get("cuda_available") is True
    egress_ok = all(r.get("ok") for r in report["egress"].values())
    write_ok = report["artifact_dir_writable"]["ok"]

    print("\n--- verdict ---", file=sys.stderr)
    print(f"GPU usable : {'yes' if gpu_ok else 'NO'}", file=sys.stderr)
    print(f"Egress ok  : {'yes' if egress_ok else 'NO'}", file=sys.stderr)
    print(f"Disk write : {'yes' if write_ok else 'NO'}", file=sys.stderr)

    return 0 if (gpu_ok and egress_ok and write_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
