"""Checks across compared interpreters."""

from __future__ import annotations

from gpu_stack_inspector.models import Finding, InterpreterFacts, Snapshot, Status, TorchFacts
from gpu_stack_inspector.rules.catalog import builtin_rules
from gpu_stack_inspector.rules.engine import evaluate

CUDA = TorchFacts("2.11.0+cu128", cuda_library=True, version_py_found=True, built_cuda="12.8")
CPU = TorchFacts("2.15.0+cpu", cuda_library=False, hip_library=False, version_py_found=True)


def env(label: str, torch: TorchFacts | None) -> InterpreterFacts:
    return InterpreterFacts(label, f"/{label}/python", version="3.12.0", torch=torch)


def diff(*interpreters: InterpreterFacts) -> list[Finding]:
    snapshot = Snapshot("0.1.0", "t", interpreters=interpreters)
    return [f for f in evaluate(snapshot, builtin_rules()) if f.id == "GSI-DIFF-001"]


def test_a_cuda_and_a_cpu_environment_are_flagged() -> None:
    [result] = diff(env("A", CUDA), env("B", CPU))
    assert (result.status, result.subject) == (Status.WARN, "comparison")
    assert (
        result.summary == "The environments have different PyTorch builds: A CUDA 12.8, B CPU only"
    )
    assert [e.value for e in result.evidence] == ["CUDA 12.8", "CPU only"]


def test_a_missing_torch_is_a_difference_too() -> None:
    [result] = diff(env("A", CUDA), env("B", None))
    assert result.summary.endswith("A CUDA 12.8, B not installed")


def test_the_same_build_everywhere_passes() -> None:
    [result] = diff(env("A", CUDA), env("B", CUDA))
    assert result.status is Status.PASS
    assert "(CUDA 12.8)" in result.summary


def test_one_interpreter_or_unusable_ones_are_not_compared() -> None:
    assert diff(env("A", CUDA)) == []
    assert diff(env("A", CUDA), InterpreterFacts("B", "/gone/python")) == []
