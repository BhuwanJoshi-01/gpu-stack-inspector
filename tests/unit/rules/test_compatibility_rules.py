"""Toolchain and driver compatibility checks."""

from __future__ import annotations

from gpu_stack_inspector.models import (
    Finding,
    InterpreterFacts,
    NvidiaFacts,
    Severity,
    Snapshot,
    Status,
    ToolkitFacts,
    TorchFacts,
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


def torch_against(
    cuda: str,
    driver_max: str | None,
    smi: str | None = "/usr/bin/nvidia-smi",
    driver: str | None = "595.99.02",
) -> Finding:
    torch = TorchFacts(
        f"2.11.0+cu{cuda.replace('.', '')}",
        cuda_library=True,
        version_py_found=True,
        built_cuda=cuda,
    )
    interpreter = InterpreterFacts("A", "/envs/a/bin/python", version="3.12.13", torch=torch)
    nvidia = NvidiaFacts(smi, driver if smi else None, driver_max)
    snapshot = Snapshot("0.1.0", "t", nvidia=nvidia, interpreters=(interpreter,))
    [result] = results(snapshot, "GSI-CUDA-003")
    return result


def test_a_torch_build_the_driver_supports_passes() -> None:
    result = torch_against("12.8", "13.2")
    assert (result.status, result.subject) == (Status.PASS, "A")
    assert result.summary == "The driver supports CUDA 12.8"


def test_a_newer_minor_build_warns_about_minor_version_compatibility() -> None:
    result = torch_against("12.8", "12.4")
    assert (result.status, result.severity) == (Status.WARN, Severity.MEDIUM)
    assert "cudaErrorCallRequiresNewerDriver" in result.reason


def test_a_newer_major_build_fails_with_the_minimum_driver() -> None:
    result = torch_against("13.0", "12.4")
    assert (result.status, result.severity) == (Status.FAIL, Severity.HIGH)
    assert "CUDA 13.x needs driver 580 or newer. CUDA will not initialise." in result.reason
    assert "CUDA 12.4 or older" in (result.recommended_action or "")
    assert result.limitations  # forward compatibility is not detected


def test_cuda_before_11_has_no_minor_version_compatibility() -> None:
    assert torch_against("10.2", "10.1").status is Status.FAIL


def test_the_build_is_not_compared_when_the_driver_did_not_answer() -> None:
    result = torch_against("12.8", None, driver=None)
    assert (result.status, result.caused_by) == (Status.NOT_TESTED, "GSI-SYS-002")


def test_cpu_builds_are_not_compared() -> None:
    torch = TorchFacts("2.15.0+cpu", cuda_library=False, hip_library=False, version_py_found=True)
    interpreter = InterpreterFacts("A", "/p", version="3.12.13", torch=torch)
    snapshot = Snapshot(
        "0.1.0",
        "t",
        nvidia=NvidiaFacts("/usr/bin/nvidia-smi", "595.99.02", "13.2"),
        interpreters=(interpreter,),
    )
    assert results(snapshot, "GSI-CUDA-003") == []
