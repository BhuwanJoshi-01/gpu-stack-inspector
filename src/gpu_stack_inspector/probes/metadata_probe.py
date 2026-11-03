"""Running the metadata probe inside a target interpreter and reading its answer."""

from __future__ import annotations

import functools
import json
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

from gpu_stack_inspector.collectors.python_env import PROBE, interpreter_path
from gpu_stack_inspector.models import Distribution, FailureKind, InterpreterFacts, ProbeFailure
from gpu_stack_inspector.runner import CommandRunner, probe_failure

PROTOCOL = 1
BEGIN, END = "GSI-PROBE-BEGIN", "GSI-PROBE-END"
TIMEOUT = 8.0
"""Starting an interpreter and reading its metadata takes about a second, even in large
environments; a hung startup (a broken .pth file, a network mount) must not hang us."""

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
        label, path, **values, distributions=tuple(distributions), failures=tuple(failures)
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


def _failed(label: str, path: str, detail: str) -> InterpreterFacts:
    return InterpreterFacts(label, path, failures=(_malformed(detail),))


def _malformed(detail: str) -> ProbeFailure:
    return ProbeFailure(PROBE, FailureKind.MALFORMED_OUTPUT, detail)
