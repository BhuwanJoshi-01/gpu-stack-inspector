"""CUDA toolkits on disk and the nvcc on PATH."""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable

from gpu_stack_inspector.models import FailureKind, ProbeFailure, ToolkitFacts
from gpu_stack_inspector.runner import CommandRunner, probe_failure

TIMEOUT = 8.0
_ENV = {"LC_ALL": "C"}
_NVCC_RELEASE = re.compile(
    r"release\s+(?P<release>[0-9]+(?:\.[0-9]+)*),\s*V(?P<full>[0-9]+(?:\.[0-9]+)*)"
)


def collect_toolkits(
    runner: CommandRunner,
    *,
    which: Callable[[str], str | None] = shutil.which,
    timeout: float = TIMEOUT,
) -> ToolkitFacts:
    """Find the nvcc on PATH and ask it for its version (``nvcc --version`` only)."""
    nvcc = which("nvcc")
    if nvcc is None:
        return ToolkitFacts()
    result = runner.run([nvcc, "--version"], timeout=timeout, env_overrides=_ENV)
    failure = probe_failure("nvcc", result)
    if failure is not None:
        return ToolkitFacts(nvcc_path=nvcc, failures=(failure,))
    version = parse_nvcc_version(result.stdout)
    failures: tuple[ProbeFailure, ...] = ()
    if version is None:
        failures = (
            ProbeFailure("nvcc", FailureKind.MALFORMED_OUTPUT, "nvcc --version printed no release"),
        )
    return ToolkitFacts(nvcc_path=nvcc, nvcc_version=version, failures=failures)


def parse_nvcc_version(stdout: str) -> str | None:
    """``Cuda compilation tools, release 12.4, V12.4.131`` → ``12.4.131``."""
    match = _NVCC_RELEASE.search(stdout)
    return match["full"] if match else None
