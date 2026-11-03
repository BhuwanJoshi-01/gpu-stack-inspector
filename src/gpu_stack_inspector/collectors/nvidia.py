"""NVIDIA driver and GPU facts from nvidia-smi."""

from __future__ import annotations

import csv
import re
import shutil
from collections.abc import Callable, Sequence

from gpu_stack_inspector.models import FailureKind, GpuDevice, NvidiaFacts, ProbeFailure
from gpu_stack_inspector.runner import CommandResult, CommandRunner, Outcome, probe_failure

PROBE = "nvidia-smi"
TIMEOUT = 8.0
"""nvidia-smi answers in well under a second; a hung driver must not hang the tool."""

_ENV = {"LC_ALL": "C"}  # decimal points and English messages, whatever the user's locale
_FIELDS = ("index", "name", "driver_version", "memory.total", "compute_cap")
_UNAVAILABLE = frozenset(
    {"", "n/a", "[n/a]", "not supported", "[not supported]", "[unknown error]"}
)
_COMPUTE_CAPABILITY = re.compile(r"[0-9]+\.[0-9]+")
_VERSION_LINE = re.compile(r"^\s*(?P<key>[A-Za-z -]+?)\s*:\s*(?P<value>\S+)\s*$", re.MULTILINE)


def collect_nvidia(
    runner: CommandRunner,
    *,
    which: Callable[[str], str | None] = shutil.which,
    timeout: float = TIMEOUT,
) -> NvidiaFacts:
    """Ask nvidia-smi for the driver version and the GPUs it can see.

    Only read-only queries are run. A missing nvidia-smi is reported as such; a failing
    one is recorded as a failure and no further queries are attempted.
    """
    smi = which("nvidia-smi")
    if smi is None:
        return NvidiaFacts(smi_path=None)

    result = runner.run(_query_argv(smi, _FIELDS), timeout=timeout, env_overrides=_ENV)
    if _no_devices(result):
        # The driver answered but sees no GPU: zero devices is the fact to report.
        versions = _versions(runner, smi, timeout)
        return NvidiaFacts(
            smi_path=smi,
            driver_version=versions.get("driver version"),
            max_cuda_version=versions.get("cuda version"),
        )
    failure = probe_failure(PROBE, result)
    if failure is not None:
        return NvidiaFacts(smi_path=smi, failures=(failure,))

    gpus, driver_version, failures = _parse_query(result.stdout, _FIELDS)
    versions = _versions(runner, smi, timeout)
    return NvidiaFacts(
        smi_path=smi,
        driver_version=driver_version or versions.get("driver version"),
        max_cuda_version=versions.get("cuda version"),
        gpus=tuple(gpus),
        failures=tuple(failures),
    )


def _versions(runner: CommandRunner, smi: str, timeout: float) -> dict[str, str]:
    """Driver and maximum CUDA version from ``nvidia-smi --version``, keys lower-cased.

    Its output looks like ``CUDA Version        : 13.2``.
    """
    result = runner.run([smi, "--version"], timeout=timeout, env_overrides=_ENV)
    if not result.ok:
        return {}
    return {
        match["key"].strip().lower(): match["value"]
        for match in _VERSION_LINE.finditer(result.stdout)
    }


def _no_devices(result: CommandResult) -> bool:
    return result.outcome is Outcome.NONZERO and "no devices were found" in result.stdout.lower()


def _query_argv(smi: str, fields: Sequence[str]) -> list[str]:
    return [smi, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]


def _parse_query(
    stdout: str, fields: Sequence[str]
) -> tuple[list[GpuDevice], str | None, list[ProbeFailure]]:
    gpus: list[GpuDevice] = []
    failures: list[ProbeFailure] = []
    driver_version: str | None = None
    try:
        rows = list(csv.reader(stdout.splitlines(), skipinitialspace=True))
    except csv.Error as exc:
        return [], None, [_malformed(f"unreadable CSV output: {exc}")]
    for number, row in enumerate(rows, start=1):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) != len(fields):
            failures.append(
                _malformed(f"line {number} has {len(row)} fields, expected {len(fields)}")
            )
            continue
        values = dict(zip(fields, (cell.strip() for cell in row), strict=True))
        index = _integer(values["index"])
        name = _value(values["name"])
        if index is None or name is None:
            failures.append(_malformed(f"line {number} has no GPU index or name"))
            continue
        gpus.append(
            GpuDevice(
                index=index,
                name=name,
                memory_total_mib=_integer(values.get("memory.total", "")),
                compute_capability=_compute_capability(values.get("compute_cap", "")),
            )
        )
        driver_version = driver_version or _value(values["driver_version"])
    return gpus, driver_version, failures


def _malformed(detail: str) -> ProbeFailure:
    return ProbeFailure(PROBE, FailureKind.MALFORMED_OUTPUT, detail)


def _value(raw: str) -> str | None:
    value = raw.strip()
    return None if value.lower() in _UNAVAILABLE else value


def _integer(raw: str) -> int | None:
    value = _value(raw)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _compute_capability(raw: str) -> str | None:
    value = _value(raw)
    return value if value is not None and _COMPUTE_CAPABILITY.fullmatch(value) else None
