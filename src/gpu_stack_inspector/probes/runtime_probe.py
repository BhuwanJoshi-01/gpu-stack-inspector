"""Running the runtime probe in a target interpreter (only with --runtime) and reading it."""

from __future__ import annotations

import functools
import json
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

from gpu_stack_inspector.models import (
    FailureKind,
    GpuDevice,
    ProbeFailure,
    RuntimeFacts,
    RuntimeStep,
)
from gpu_stack_inspector.probes.metadata_probe import python_error_line
from gpu_stack_inspector.runner import CommandResult, CommandRunner, Outcome, probe_failure

PROBE = "runtime"
TIMEOUT = 60.0
"""Importing torch and creating a CUDA context takes a few seconds (2-4 s on the test
laptop); a driver that hangs must not hang the tool."""

BEGIN, END, STEP = "GSI-RUNTIME-BEGIN", "GSI-RUNTIME-END", "GSI-RUNTIME-STEP"
_SIGNALS = {
    -11: "a segmentation fault (SIGSEGV)",
    139: "a segmentation fault (SIGSEGV)",
    -6: "an abort (SIGABRT)",
    134: "an abort (SIGABRT)",
    -9: "being killed (SIGKILL)",
    137: "being killed (SIGKILL)",
    -7: "a bus error (SIGBUS)",
    135: "a bus error (SIGBUS)",
}


@functools.cache
def program_source() -> str:
    resource = files("gpu_stack_inspector.probes").joinpath("runtime_program.py")
    return resource.read_text(encoding="utf-8")


def run_runtime_probe(
    runner: CommandRunner, path: str, *, timeout: float = TIMEOUT
) -> RuntimeFacts:
    """Import torch in the interpreter at ``path`` and use CUDA once. Only with consent."""
    with tempfile.TemporaryDirectory(prefix="gsi-runtime-") as empty:
        result = runner.run([path, "-c", program_source()], timeout=timeout, cwd=Path(empty))
    announced = [
        line[len(STEP) + 1 :].strip()
        for line in result.stdout.splitlines()
        if line.startswith(STEP + " ")
    ]
    if result.outcome is Outcome.TIMEOUT:
        where = f" in step '{announced[-1]}'" if announced else ""
        detail = (
            f"no answer within {result.duration:.0f} s{where}; "
            "CUDA initialisation can hang on a broken driver"
        )
        return _died(announced, ProbeFailure(PROBE, FailureKind.TIMEOUT, detail))
    if result.outcome is Outcome.NONZERO:
        return _died(announced, _crash(result, announced))
    failure = probe_failure(PROBE, result)
    if failure is not None:
        return RuntimeFacts(failure=failure)
    return parse_runtime_output(result.stdout)


def _crash(result: CommandResult, announced: list[str]) -> ProbeFailure:
    signal = _SIGNALS.get(result.returncode or 0)
    how = f"crashed with {signal}" if signal else f"exited with code {result.returncode}"
    where = f" during '{announced[-1]}'" if announced else ""
    reason = python_error_line(result.stderr)
    detail = f"the interpreter {how}{where}" + (f": {reason}" if reason and not signal else "")
    return ProbeFailure(PROBE, FailureKind.NONZERO_EXIT, detail)


def _died(announced: list[str], failure: ProbeFailure) -> RuntimeFacts:
    """Steps reconstructed from the markers: all finished but the last, which it died in."""
    steps = tuple(
        RuntimeStep(
            name,
            ok=index < len(announced) - 1,
            error=None if index < len(announced) - 1 else failure.detail,
        )
        for index, name in enumerate(announced)
    )
    return RuntimeFacts(steps=steps, failure=failure)


def parse_runtime_output(stdout: str) -> RuntimeFacts:
    start = stdout.find(BEGIN)
    end = stdout.find(END, start + len(BEGIN)) if start != -1 else -1
    if start == -1 or end == -1:
        return RuntimeFacts(failure=_malformed("the interpreter printed no runtime result"))
    try:
        document = json.loads(stdout[start + len(BEGIN) : end])
    except ValueError:
        return RuntimeFacts(failure=_malformed("the runtime result is not valid JSON"))
    if not isinstance(document, dict) or document.get("protocol") != 1:
        return RuntimeFacts(failure=_malformed("unexpected runtime probe protocol"))
    steps: list[RuntimeStep] = []
    values: dict[str, Any] = {}
    for raw in document.get("steps") or []:
        if not isinstance(raw, dict) or not isinstance(raw.get("step"), str):
            continue
        seconds = raw.get("seconds")
        error = raw.get("error")
        steps.append(
            RuntimeStep(
                raw["step"],
                ok=raw.get("ok") is True,
                seconds=float(seconds) if isinstance(seconds, int | float) else None,
                error=error if isinstance(error, str) else None,
            )
        )
        if raw.get("ok") is True:
            values[raw["step"]] = raw.get("value")
    imported = values.get("import torch")
    imported = imported if isinstance(imported, dict) else {}
    available = values.get("cuda available")
    attempted = any(s.step == "tensor operation" for s in steps)
    return RuntimeFacts(
        torch_version=_string(imported.get("version")),
        cuda_version=_string(imported.get("cuda")),
        cuda_available=available if isinstance(available, bool) else None,
        devices=_devices(values.get("devices")),
        tensor_ok=("tensor operation" in values) if attempted else None,
        steps=tuple(steps),
    )


def _devices(raw: object) -> tuple[GpuDevice, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(
        GpuDevice(item["index"], item["name"], None, _string(item.get("capability")))
        for item in raw
        if isinstance(item, dict)
        and isinstance(item.get("index"), int)
        and isinstance(item.get("name"), str)
    )


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _malformed(detail: str) -> ProbeFailure:
    return ProbeFailure(PROBE, FailureKind.MALFORMED_OUTPUT, detail)
