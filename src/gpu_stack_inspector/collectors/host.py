"""Operating system facts."""

from __future__ import annotations

import platform
from collections.abc import Callable

from gpu_stack_inspector.models import HostFacts


def collect_host(
    *,
    system: Callable[[], str] = platform.system,
    machine: Callable[[], str] = platform.machine,
    release: Callable[[], str] = platform.release,
) -> HostFacts:
    """Read the OS name, architecture and kernel release; never runs a command."""
    kernel = release() or None
    return HostFacts(
        system=system() or "unknown",
        machine=machine() or "unknown",
        kernel=kernel,
        wsl=_is_wsl(kernel),
    )


def _is_wsl(kernel: str | None) -> bool:
    # WSL kernels are built by Microsoft and say so: "5.15.167.4-microsoft-standard-WSL2".
    return kernel is not None and "microsoft" in kernel.lower()
