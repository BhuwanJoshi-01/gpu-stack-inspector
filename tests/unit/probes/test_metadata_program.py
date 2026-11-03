"""The program that runs inside target interpreters."""

from __future__ import annotations

import ast
import json
import sys
from importlib.resources import files

from gpu_stack_inspector.runner import Outcome, SubprocessRunner

SOURCE = files("gpu_stack_inspector.probes").joinpath("metadata_program.py").read_text()
STDLIB_ONLY = {
    "__future__",
    "json",
    "os",
    "site",
    "sys",
    "sysconfig",
    "re",
    "importlib",
    "importlib.metadata",
}


def run_probe(**kwargs: object) -> dict[str, object]:
    result = SubprocessRunner().run([sys.executable, "-c", SOURCE], timeout=30, **kwargs)  # type: ignore[arg-type]
    assert result.outcome is Outcome.OK, result.stderr
    begin, end = result.stdout.index("GSI-PROBE-BEGIN"), result.stdout.index("GSI-PROBE-END")
    document: dict[str, object] = json.loads(result.stdout[begin + len("GSI-PROBE-BEGIN") : end])
    return document


def test_the_program_is_valid_python_3_8() -> None:
    ast.parse(SOURCE, feature_version=(3, 8))


def test_the_program_imports_only_the_standard_library() -> None:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported <= STDLIB_ONLY, imported - STDLIB_ONLY


def test_the_program_reports_the_interpreter_it_runs_in() -> None:
    document = run_probe()
    assert document["protocol"] == 1
    python = document["python"]
    assert isinstance(python, dict)
    assert python["version"] == ".".join(str(part) for part in sys.version_info[:3])
    assert python["executable"] == sys.executable
    assert python["prefix"] == sys.prefix
    assert python["base_prefix"] == sys.base_prefix
    assert python["implementation"] == sys.implementation.name
    assert isinstance(python["conda"], bool)


def test_pythonpath_is_reported_as_set_or_not() -> None:
    assert (
        run_probe(env_overrides={"PYTHONPATH": "/nonexistent"})["python"]["pythonpath_set"] is True
    )  # type: ignore[index]
