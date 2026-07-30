"""The five adversarial scenarios.

Each scenario is split in two halves that never share a process:

  `cause`       runs INSIDE the sandbox and deliberately breaks something.
  `expectations` describe what the PLATFORM must do about it, checked from
                outside by conformance.observer.

The split is not stylistic. A process that gets OOM-killed cannot report on how
its own death was handled, and a filesystem filled to 100% cannot be relied on
to write a result file. Any design where the failing job grades itself is
measuring nothing.
"""

from __future__ import annotations

from .cuda_oom import SCENARIO as CUDA_OOM
from .disk_fill import SCENARIO as DISK_FILL
from .host_ram import SCENARIO as HOST_RAM
from .hung_network import SCENARIO as HUNG_NETWORK
from .kill_resume import SCENARIO as KILL_RESUME

ALL_SCENARIOS = (CUDA_OOM, HOST_RAM, HUNG_NETWORK, KILL_RESUME, DISK_FILL)
BY_NAME = {s.name: s for s in ALL_SCENARIOS}

__all__ = [
    "ALL_SCENARIOS",
    "BY_NAME",
    "CUDA_OOM",
    "DISK_FILL",
    "HOST_RAM",
    "HUNG_NETWORK",
    "KILL_RESUME",
]
