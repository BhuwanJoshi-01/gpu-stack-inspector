"""Interpreter checks on hand-made snapshots and real probe answers."""

from __future__ import annotations

from dataclasses import replace

from gpu_stack_inspector.models import (
    Distribution,
    FailureKind,
    Finding,
    InterpreterFacts,
    ProbeFailure,
    Snapshot,
    Status,
    TorchFacts,
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


def test_an_interpreter_from_another_environment_than_the_shell_is_noted() -> None:
    context = RuleContext(shell_python="/usr/bin/python3")
    [result] = results(HEALTHY, rule="GSI-ENV-002", context=context)
    assert (result.status, result.subject) == (Status.WARN, "A")
    assert "/usr/bin/python3" in result.reason


def test_the_shells_own_environment_is_not_noted() -> None:
    context = RuleContext(shell_python="/envs/a/bin/python3")
    assert results(HEALTHY, rule="GSI-ENV-002", context=context) == []


def test_nothing_is_noted_when_the_shell_python_is_unknown_or_the_interpreter_unusable() -> None:
    assert results(HEALTHY, rule="GSI-ENV-002") == []
    missing = InterpreterFacts("B", "/gone/bin/python")
    assert (
        results(missing, rule="GSI-ENV-002", context=RuleContext(shell_python="/usr/bin/python3"))
        == []
    )


def with_torch(version: str = "2.11.0+cu128", **fields: object) -> InterpreterFacts:
    torch = TorchFacts(version=version, location="/envs/a/lib/site-packages", **fields)  # type: ignore[arg-type]
    return replace(HEALTHY, torch=torch)


def test_installed_torch_passes() -> None:
    [result] = results(with_torch(), rule="GSI-TORCH-001")
    assert result.status is Status.PASS
    assert result.summary == "torch 2.11.0+cu128 is installed"


def test_missing_torch_with_packages_that_need_it_warns() -> None:
    env = replace(
        HEALTHY, distributions=(Distribution("accelerate", "1.13.0"), Distribution("numpy", "2.4"))
    )
    [result] = results(env, rule="GSI-TORCH-001")
    assert result.status is Status.WARN
    assert result.reason == "accelerate cannot work without torch."
    assert "pytorch.org" in (result.recommended_action or "")


def test_an_environment_simply_without_torch_is_not_applicable() -> None:
    [result] = results(HEALTHY, rule="GSI-TORCH-001")
    assert result.status is Status.NOT_APPLICABLE
