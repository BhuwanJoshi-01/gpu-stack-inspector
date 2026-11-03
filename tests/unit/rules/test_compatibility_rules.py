"""Toolchain and driver compatibility checks."""

from __future__ import annotations

from gpu_stack_inspector.models import (
    Finding,
    NvidiaFacts,
    Severity,
    Snapshot,
    Status,
    ToolkitFacts,
)
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


def nvcc_against(
    nvcc: str, driver_max: str | None, smi: str | None = "/usr/bin/nvidia-smi"
) -> Finding:
    nvidia = NvidiaFacts(smi, "595.99.02" if smi else None, driver_max)
    snapshot = Snapshot("0.1.0", "t", nvidia=nvidia, toolkit=ToolkitFacts("/usr/bin/nvcc", nvcc))
    [result] = results(snapshot, "GSI-CUDA-002")
    return result


def test_nvcc_older_than_the_driver_passes() -> None:
    result = nvcc_against("12.4.131", "13.2")
    assert (result.status, result.summary) == (Status.PASS, "The driver supports nvcc's CUDA 12.4")


def test_nvcc_of_a_newer_minor_version_warns_about_ptx() -> None:
    result = nvcc_against("12.8.93", "12.4")
    assert (result.status, result.severity) == (Status.WARN, Severity.LOW)
    assert "PTX will not" in result.reason


def test_nvcc_of_a_newer_major_version_warns_with_the_minimum_driver() -> None:
    result = nvcc_against("13.0.88", "12.4")
    assert (result.status, result.severity) == (Status.WARN, Severity.MEDIUM)
    assert "CUDA 13.x needs driver 580 or newer." in result.reason


def test_nvcc_is_not_compared_when_the_driver_is_unknown() -> None:
    result = nvcc_against("12.4.131", None, smi=None)
    assert (result.status, result.caused_by) == (Status.NOT_TESTED, "GSI-SYS-001")
