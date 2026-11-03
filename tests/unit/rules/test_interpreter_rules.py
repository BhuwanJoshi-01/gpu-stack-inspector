"""Interpreter checks on hand-made snapshots and real probe answers."""

from __future__ import annotations

from gpu_stack_inspector.models import (
    FailureKind,
    Finding,
    InterpreterFacts,
    ProbeFailure,
    Snapshot,
    Status,
)
from gpu_stack_inspector.rules.catalog import builtin_rules
from gpu_stack_inspector.rules.engine import RuleContext, evaluate

HEALTHY = InterpreterFacts(
    "A",
    "/envs/a/bin/python",
    version="3.12.13",
    implementation="cpython",
    conda=True,
    prefix="/envs/a",
    base_prefix="/envs/a",
)


def results(
    *interpreters: InterpreterFacts, rule: str, context: RuleContext | None = None
) -> list[Finding]:
    snapshot = Snapshot("0.1.0", "t", interpreters=interpreters)
    return [f for f in evaluate(snapshot, builtin_rules(), context) if f.id == rule]


def test_a_healthy_interpreter_passes_with_its_kind() -> None:
    [result] = results(HEALTHY, rule="GSI-ENV-001")
    assert (result.status, result.subject) == (Status.PASS, "A")
    assert result.summary == "Python 3.12.13 (cpython, conda env)"


def test_a_partly_inspected_interpreter_passes_with_limitations() -> None:
    partial = InterpreterFacts(
        "A",
        "/p",
        version="3.12.0",
        failures=(ProbeFailure("python", FailureKind.PROBE_ERROR, "distributions: OSError"),),
    )
    [result] = results(partial, rule="GSI-ENV-001")
    assert result.status is Status.PASS
    assert result.limitations == ("distributions: OSError",)


def test_an_interpreter_that_cannot_be_started_fails_with_path_advice() -> None:
    missing = InterpreterFacts(
        "B",
        "/gone/bin/python",
        failures=(
            ProbeFailure("python", FailureKind.NOT_FOUND, "/gone/bin/python does not exist"),
        ),
    )
    [result] = results(missing, rule="GSI-ENV-001")
    assert result.status is Status.FAIL
    assert result.summary == "The interpreter cannot be started"
    assert result.reason == "/gone/bin/python does not exist"
    assert "moved or deleted" in (result.recommended_action or "")
    assert result.verify_command == "/gone/bin/python --version"


def test_an_interpreter_that_crashes_fails_with_different_advice() -> None:
    crashed = InterpreterFacts(
        "C",
        "/broken/bin/python",
        failures=(
            ProbeFailure(
                "python",
                FailureKind.NONZERO_EXIT,
                "/broken/bin/python exited with code 1: "
                "ModuleNotFoundError: No module named 'encodings'",
            ),
        ),
    )
    [result] = results(crashed, rule="GSI-ENV-001")
    assert result.summary == "The interpreter started but did not answer"
    assert "encodings" in result.reason
    assert "recreated" in (result.recommended_action or "")
