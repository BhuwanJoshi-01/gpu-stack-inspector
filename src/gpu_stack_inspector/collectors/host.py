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
    os_release: Callable[[], dict[str, str]] = platform.freedesktop_os_release,
) -> HostFacts:
    """Read the OS name, architecture, kernel release and Linux distribution.

    Never runs a command. The distribution comes from ``/etc/os-release`` (or
    ``/usr/lib/os-release``); if neither can be read it stays unknown.
    """
    name = system() or "unknown"
    kernel = release() or None
    return HostFacts(
        system=name,
        machine=machine() or "unknown",
        kernel=kernel,
        distribution=_distribution(os_release) if name == "Linux" else None,
        wsl=_is_wsl(kernel),
    )


def _distribution(os_release: Callable[[], dict[str, str]]) -> str | None:
    try:
        info = os_release()
    except OSError:
        return None
    pretty = info.get("PRETTY_NAME", "").strip()
    if pretty:
        return pretty
    fallback = " ".join(part for part in (info.get("NAME", ""), info.get("VERSION_ID", "")) if part)
    return fallback or None


def _is_wsl(kernel: str | None) -> bool:
    # WSL kernels are built by Microsoft and say so: "5.15.167.4-microsoft-standard-WSL2".
    return kernel is not None and "microsoft" in kernel.lower()
