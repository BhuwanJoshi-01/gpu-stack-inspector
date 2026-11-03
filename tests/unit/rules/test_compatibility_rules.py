"""Toolchain and driver compatibility checks."""

from __future__ import annotations

from gpu_stack_inspector.models import Finding, Snapshot, Status, ToolkitFacts
from gpu_stack_inspector.rules.catalog import builtin_rules
from gpu_stack_inspector.rules.engine import evaluate


def results(snapshot: Snapshot, rule: str) -> list[Finding]:
    return [f for f in evaluate(snapshot, builtin_rules()) if f.id == rule]


def test_an_installed_nvcc_passes() -> None:
    snapshot = Snapshot("0.1.0", "t", toolkit=ToolkitFacts("/usr/bin/nvcc", "12.4.131"))
    [result] = results(snapshot, "GSI-CUDA-001")
    assert (result.status, result.summary) == (Status.PASS, "nvcc 12.4.131 is installed")


def test_no_nvcc_is_not_a_problem_for_prebuilt_wheels() -> None:
    [result] = results(Snapshot("0.1.0", "t", toolkit=ToolkitFacts()), "GSI-CUDA-001")
    assert result.status is Status.NOT_APPLICABLE
    assert "Prebuilt PyTorch wheels do not need it" in result.reason
