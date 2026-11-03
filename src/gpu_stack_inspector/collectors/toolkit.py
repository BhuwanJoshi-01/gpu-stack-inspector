"""CUDA toolkits on disk and the nvcc on PATH."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

from gpu_stack_inspector.models import CudaToolkit, FailureKind, ProbeFailure, ToolkitFacts
from gpu_stack_inspector.runner import CommandRunner, probe_failure

TIMEOUT = 8.0
DEFAULT_ROOTS: tuple[Path, ...] = () if os.name == "nt" else (Path("/usr/local"), Path("/opt"))
"""Where toolkits are installed: /usr/local/cuda-12.4 (NVIDIA's installer), /opt/cuda (Arch)."""

_MAX_FILE_BYTES = 64 * 1024
_TOOLKIT_DIR = re.compile(r"cuda(?:-[0-9]+(?:\.[0-9]+)*)?")
_VERSION_TXT = re.compile(r"CUDA Version\s+(?P<version>[0-9]+(?:\.[0-9]+)+)")
_CUDA_H = re.compile(r"^\s*#\s*define\s+CUDA_VERSION\s+(?P<number>[0-9]{4,6})\b", re.MULTILINE)
_ENV = {"LC_ALL": "C"}
_NVCC_RELEASE = re.compile(
    r"release\s+(?P<release>[0-9]+(?:\.[0-9]+)*),\s*V(?P<full>[0-9]+(?:\.[0-9]+)*)"
)


def collect_toolkits(
    runner: CommandRunner,
    *,
    which: Callable[[str], str | None] = shutil.which,
    roots: Sequence[Path] = DEFAULT_ROOTS,
    timeout: float = TIMEOUT,
) -> ToolkitFacts:
    """Find CUDA toolkit directories and the nvcc on PATH.

    Directories are only listed and small version files read; the only command run is
    ``nvcc --version``.
    """
    toolkits = tuple(find_toolkits(roots))
    nvcc_facts = _collect_nvcc(runner, which, timeout)
    return ToolkitFacts(
        nvcc_path=nvcc_facts.nvcc_path,
        nvcc_version=nvcc_facts.nvcc_version,
        toolkits=toolkits,
        failures=nvcc_facts.failures,
    )


def find_toolkits(roots: Sequence[Path]) -> list[CudaToolkit]:
    """``cuda`` and ``cuda-X.Y`` directories directly under each root, with their version."""
    toolkits: list[CudaToolkit] = []
    for root in roots:
        try:
            entries = sorted(root.iterdir())
        except OSError:  # a missing or unreadable root is normal
            continue
        for entry in entries:
            if _TOOLKIT_DIR.fullmatch(entry.name) and entry.is_dir():
                version, source = read_toolkit_version(entry)
                toolkits.append(CudaToolkit(str(entry), version, source))
    return toolkits


def read_toolkit_version(directory: Path) -> tuple[str | None, str | None]:
    """The toolkit's version and the file it came from, or ``(None, None)``."""
    text = _read_small(directory / "version.json")
    if text is not None:
        try:
            version = json.loads(text)["cuda"]["version"]
        except (ValueError, KeyError, TypeError):
            version = None
        if isinstance(version, str) and version:
            return version, "version.json"
    text = _read_small(directory / "version.txt")  # CUDA 10.x and older
    match = _VERSION_TXT.search(text) if text is not None else None
    if match:
        return match["version"], "version.txt"
    text = _read_small(directory / "include" / "cuda.h")
    match = _CUDA_H.search(text) if text is not None else None
    if match:
        # CUDA_VERSION is 1000 * major + 10 * minor: 12040 is 12.4.
        number = int(match["number"])
        return f"{number // 1000}.{number % 1000 // 10}", "include/cuda.h"
    return None, None


def _read_small(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            return handle.read(_MAX_FILE_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return None


def _collect_nvcc(
    runner: CommandRunner, which: Callable[[str], str | None], timeout: float
) -> ToolkitFacts:
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
