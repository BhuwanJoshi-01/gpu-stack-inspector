"""CUDA toolkits on disk and the nvcc on PATH."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from gpu_stack_inspector.models import CudaToolkit, FailureKind, ProbeFailure, ToolkitFacts
from gpu_stack_inspector.runner import CommandRunner, probe_failure

TIMEOUT = 8.0
DEFAULT_ROOTS: tuple[Path, ...] = () if os.name == "nt" else (Path("/usr/local"), Path("/opt"))
"""Where toolkits are installed: /usr/local/cuda-12.4 (NVIDIA's installer), /opt/cuda (Arch)."""

_MAX_FILE_BYTES = 64 * 1024
_PROBE = "cuda-toolkits"
_ENV_DIRS = ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT")
"""Variables naming a toolkit; Windows also sets one CUDA_PATH_V<major>_<minor> per version."""
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
    environ: Mapping[str, str] = os.environ,
    timeout: float = TIMEOUT,
) -> ToolkitFacts:
    """Find CUDA toolkit directories and the nvcc on PATH.

    Directories are only listed and small version files read; the only command run is
    ``nvcc --version``.
    """
    toolkits, dir_failures = find_toolkits(roots, environ)
    nvcc_facts = _collect_nvcc(runner, which, timeout)
    return ToolkitFacts(
        nvcc_path=nvcc_facts.nvcc_path,
        nvcc_version=nvcc_facts.nvcc_version,
        toolkits=tuple(toolkits),
        failures=(*dir_failures, *nvcc_facts.failures),
    )


def find_toolkits(
    roots: Sequence[Path], environ: Mapping[str, str] | None = None
) -> tuple[list[CudaToolkit], list[ProbeFailure]]:
    """Toolkit directories under each root and named by CUDA_HOME/CUDA_PATH/….

    Each directory is reported once under its real path; symlinks to it, such as
    ``/usr/local/cuda``, are listed in ``linked_from``. A variable that names a missing
    directory is a failure worth reporting: tools that read it will not find CUDA.
    """
    candidates: list[Path] = []
    for root in roots:
        try:
            entries = sorted(root.iterdir())
        except OSError:  # a missing or unreadable root is normal
            continue
        candidates.extend(entry for entry in entries if _TOOLKIT_DIR.fullmatch(entry.name))

    failures: list[ProbeFailure] = []
    env = environ or {}
    names = [*_ENV_DIRS, *sorted(n for n in env if n.startswith("CUDA_PATH_V"))]
    for name in names:
        value = env.get(name, "")
        if not value:
            continue
        if Path(value).is_dir():
            candidates.append(Path(value))
        else:
            failures.append(
                ProbeFailure(
                    _PROBE, FailureKind.NOT_FOUND, f"{name} names {value}, which is not a directory"
                )
            )

    aliases: dict[Path, list[str]] = {}
    for candidate in candidates:
        try:
            target = candidate.resolve(strict=True)
        except OSError:  # dangling symlink
            continue
        if not target.is_dir():
            continue
        links = aliases.setdefault(target, [])
        if candidate.is_symlink() and str(candidate) not in links:
            links.append(str(candidate))

    toolkits = []
    for target, links in aliases.items():
        version, source = read_toolkit_version(target)
        toolkits.append(CudaToolkit(str(target), version, source, tuple(links)))
    return toolkits, failures


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
