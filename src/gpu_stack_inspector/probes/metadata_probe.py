"""Running the metadata probe inside a target interpreter and reading its answer."""

from __future__ import annotations

import functools
import json
import re
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

from gpu_stack_inspector.collectors.python_env import PROBE, interpreter_path
from gpu_stack_inspector.models import (
    Distribution,
    FailureKind,
    InterpreterFacts,
    ProbeFailure,
    TorchFacts,
)
from gpu_stack_inspector.runner import CommandResult, CommandRunner, Outcome, probe_failure

PROTOCOL = 1
BEGIN, END = "GSI-PROBE-BEGIN", "GSI-PROBE-END"
TIMEOUT = 8.0
"""Starting an interpreter and reading its metadata takes about a second, even in large
environments; a hung startup (a broken .pth file, a network mount) must not hang us."""

_EXCEPTION_LINE = re.compile(r"[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt)\b.*")
_DETAIL_LIMIT = 300

_STRING_FIELDS = ("version", "implementation", "executable", "prefix", "base_prefix", "platform")
_BOOL_FIELDS = ("conda", "user_site", "pythonpath_set")


@functools.cache
def program_source() -> str:
    """The probe program, passed to the target as ``python -c``."""
    resource = files("gpu_stack_inspector.probes").joinpath("metadata_program.py")
    return resource.read_text(encoding="utf-8")


def probe_interpreter(
    runner: CommandRunner,
    label: str,
    raw_path: str,
    *,
    cwd: Path | None = None,
    timeout: float = TIMEOUT,
) -> InterpreterFacts:
    """Run the metadata probe in the interpreter named by ``raw_path``.

    The target's environment is passed on unchanged: its variables are part of what is
    being diagnosed. Only the working directory is replaced, see below.
    """
    path, failure = interpreter_path(raw_path, cwd=cwd)
    if path is None:
        return InterpreterFacts(label, raw_path, failures=(failure,) if failure else ())
    argv = [str(path), "-c", program_source()]
    # With -c, Python puts the working directory first on sys.path. Running in a new
    # empty directory means a module planted where the user runs the tool (a json.py,
    # say) is never imported by the probe.
    with tempfile.TemporaryDirectory(prefix="gsi-probe-") as empty:
        result = runner.run(argv, timeout=timeout, cwd=Path(empty))
    if result.outcome is Outcome.NONZERO:
        return InterpreterFacts(label, str(path), failures=(_crash(path, result),))
    failure = probe_failure(PROBE, result)
    if failure is not None:
        return InterpreterFacts(label, str(path), failures=(failure,))
    return parse_probe_output(label, str(path), result.stdout, truncated=result.stdout_truncated)


def parse_probe_output(
    label: str, path: str, stdout: str, *, truncated: bool = False
) -> InterpreterFacts:
    """Read the JSON document between the markers; anything unexpected becomes unknown."""
    start = stdout.find(BEGIN)
    end = stdout.find(END, start + len(BEGIN)) if start != -1 else -1
    if start == -1 or end == -1:
        if truncated:
            return _failed(
                label, path, "the probe's output exceeded the output limit and was cut off"
            )
        return _failed(label, path, "the interpreter printed no probe result")
    try:
        document = json.loads(stdout[start + len(BEGIN) : end])
    except ValueError:
        return _failed(label, path, "the probe result is not valid JSON")
    if not isinstance(document, dict):
        return _failed(label, path, "the probe result is not a JSON object")
    if document.get("protocol") != PROTOCOL:
        return _failed(label, path, f"unexpected probe protocol {document.get('protocol')!r}")

    unexpected: list[str] = []
    values = _python_values(document.get("python"), unexpected)
    distributions = _distributions(document.get("distributions"), unexpected)
    torch = _torch(document.get("torch"), unexpected)
    failures: list[ProbeFailure] = []
    if unexpected:
        detail = "unexpected values for: " + ", ".join(dict.fromkeys(unexpected))
        failures.append(_malformed(detail))
    errors = document.get("errors")
    if isinstance(errors, list):
        failures.extend(
            ProbeFailure(PROBE, FailureKind.PROBE_ERROR, error)
            for error in errors
            if isinstance(error, str)
        )
    return InterpreterFacts(
        label,
        path,
        **values,
        distributions=tuple(distributions),
        torch=torch,
        failures=tuple(failures),
    )


def _python_values(section: object, unexpected: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = dict.fromkeys((*_STRING_FIELDS, *_BOOL_FIELDS))
    if section is None:
        return values
    if not isinstance(section, dict):
        unexpected.append("python")
        return values
    for keys, kind in ((_STRING_FIELDS, str), (_BOOL_FIELDS, bool)):
        for key in keys:
            value = section.get(key)
            if value is None or isinstance(value, kind):
                values[key] = value
            else:
                unexpected.append(key)
    return values


def _distributions(section: object, unexpected: list[str]) -> list[Distribution]:
    if section is None:
        return []
    if not isinstance(section, list):
        unexpected.append("distributions")
        return []
    found = []
    for entry in section:
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("name"), str)
            and isinstance(entry.get("version"), str)
            and isinstance(entry.get("location"), str | None)
        ):
            found.append(Distribution(entry["name"], entry["version"], entry.get("location")))
        else:
            unexpected.append("distributions")
    return found


def python_error_line(stderr: str) -> str:
    """The line of a Python crash that names the problem, or "" if there is none.

    The exception at the end of a traceback, else the "Fatal Python error" line, else the
    first line of stderr.
    """
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    reason = next((line for line in reversed(lines) if _EXCEPTION_LINE.fullmatch(line)), None)
    if reason is None:
        reason = next((line for line in lines if line.startswith("Fatal Python error")), None)
    if reason is None:
        reason = lines[0] if lines else ""
    return reason


def _crash(path: Path, result: CommandResult) -> ProbeFailure:
    """Explain a crashed interpreter by the line that names the problem.

    A traceback starts with "Traceback (most recent call last):", which says nothing;
    the exception is on its last line. "Fatal Python error" output (a broken standard
    library, for instance) names the underlying exception further down.
    """
    reason = python_error_line(result.stderr)
    detail = f"{path} exited with code {result.returncode}"
    if reason:
        detail += f": {reason}"
    if reason.startswith("SyntaxError"):
        detail += " (the probe needs Python 3.8 or newer)"
    if len(detail) > _DETAIL_LIMIT:
        detail = detail[: _DETAIL_LIMIT - 1] + "…"
    return ProbeFailure(PROBE, FailureKind.NONZERO_EXIT, detail)


def _torch(section: object, unexpected: list[str]) -> TorchFacts | None:
    if section is None:
        return None
    if not isinstance(section, dict) or not isinstance(section.get("version"), str):
        unexpected.append("torch")
        return None

    def optional(value: object, kind: type, key: str) -> Any:
        if value is None or isinstance(value, kind):
            return value
        unexpected.append(f"torch.{key}")
        return None

    def strings(value: object, key: str) -> tuple[str, ...] | None:
        if value is None:
            return None
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return tuple(value)
        unexpected.append(f"torch.{key}")
        return None

    libraries = optional(section.get("libraries"), dict, "libraries") or {}
    version_py = optional(section.get("version_py"), dict, "version_py")
    return TorchFacts(
        version=section["version"],
        location=optional(section.get("location"), str, "location"),
        requires_nvidia=strings(section.get("requires_nvidia"), "requires_nvidia") or (),
        cuda_library=optional(libraries.get("cuda"), bool, "libraries.cuda"),
        hip_library=optional(libraries.get("hip"), bool, "libraries.hip"),
        version_py_found=version_py is not None,
        built_cuda=optional((version_py or {}).get("cuda"), str, "version_py.cuda"),
        built_hip=optional((version_py or {}).get("hip"), str, "version_py.hip"),
        conda_records=strings(section.get("conda_records"), "conda_records"),
    )


def _failed(label: str, path: str, detail: str) -> InterpreterFacts:
    return InterpreterFacts(label, path, failures=(_malformed(detail),))


def _malformed(detail: str) -> ProbeFailure:
    return ProbeFailure(PROBE, FailureKind.MALFORMED_OUTPUT, detail)
