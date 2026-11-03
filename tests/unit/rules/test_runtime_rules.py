"""Runtime checks: CUDA proven to work, or why not."""

from __future__ import annotations

from dataclasses import replace

from gpu_stack_inspector.models import (
    EnvironmentVariable,
    FailureKind,
    Finding,
    GpuDevice,
    InterpreterFacts,
    NvidiaFacts,
    ProbeFailure,
    RuntimeFacts,
    RuntimeStep,
    Snapshot,
    Status,
    TorchFacts,
)
from gpu_stack_inspector.probes.runtime_probe import parse_runtime_output
from gpu_stack_inspector.rules.catalog import builtin_rules
from gpu_stack_inspector.rules.engine import evaluate
from tests.fakes import read_fixture

CUDA_TORCH = TorchFacts("2.11.0+cu128", cuda_library=True, version_py_found=True, built_cuda="12.8")
CPU_TORCH = TorchFacts("2.15.0+cpu", cuda_library=False, hip_library=False, version_py_found=True)
DRIVER = NvidiaFacts(
    "/usr/bin/nvidia-smi", "595.99.02", "13.2", (GpuDevice(0, "RTX 3050", 4096, "8.6"),)
)
ENV = InterpreterFacts("A", "/envs/a/bin/python", version="3.13.13", torch=CUDA_TORCH)


def results(interpreter: InterpreterFacts, rule: str, **snapshot_fields: object) -> list[Finding]:
    fields = {"nvidia": DRIVER, **snapshot_fields}
    snapshot = Snapshot("0.1.0", "t", interpreters=(interpreter,), **fields)  # type: ignore[arg-type]
    return [f for f in evaluate(snapshot, builtin_rules()) if f.id == rule]


def runtime(fixture: str) -> RuntimeFacts:
    return parse_runtime_output(read_fixture("runtime", fixture))


def test_a_cuda_operation_that_ran_passes_with_the_device() -> None:
    [result] = results(replace(ENV, runtime=runtime("sapiens2.stdout.txt")), "GSI-TORCH-003")
    assert result.status is Status.PASS
    assert result.summary == "A CUDA operation ran on NVIDIA GeForce RTX 3050 Laptop GPU"
    assert "torch 2.11.0+cu128 with CUDA 12.8" in result.reason


def test_cuda_unavailable_in_a_cpu_build_follows_from_torch_002() -> None:
    env = replace(ENV, torch=CPU_TORCH, runtime=runtime("cpu-venv.stdout.txt"))
    [result] = results(env, "GSI-TORCH-003")
    assert result.status is Status.FAIL
    assert result.summary == "torch.cuda.is_available() is False"
    assert result.caused_by == "GSI-TORCH-002"


def test_cuda_unavailable_because_of_cuda_visible_devices_follows_from_env_003() -> None:
    unavailable = RuntimeFacts(torch_version="2.11.0+cu128", cuda_available=False)
    hidden = (EnvironmentVariable("CUDA_VISIBLE_DEVICES", ""),)
    [result] = results(replace(ENV, runtime=unavailable), "GSI-TORCH-003", cuda_environment=hidden)
    assert result.caused_by == "GSI-ENV-003"


def test_a_missing_kernel_image_names_the_compute_capability() -> None:
    failed = RuntimeFacts(
        torch_version="2.11.0+cu128",
        cuda_available=True,
        devices=(GpuDevice(0, "Tesla K80", None, "3.7"),),
        tensor_ok=False,
        steps=(
            RuntimeStep("import torch", True),
            RuntimeStep("cuda available", True),
            RuntimeStep("devices", True),
            RuntimeStep(
                "tensor operation",
                False,
                error=(
                    "RuntimeError: CUDA error: no kernel image is available for execution "
                    "on the device"
                ),
            ),
        ),
    )
    [result] = results(replace(ENV, runtime=failed), "GSI-TORCH-003")
    assert (result.status, result.summary) == (Status.FAIL, "A CUDA operation failed")
    assert "compute capability 3.7" in (result.recommended_action or "")


def test_a_crash_while_using_cuda_fails_with_the_detail() -> None:
    detail = "the interpreter crashed with a segmentation fault (SIGSEGV) during 'cuda available'"
    crashed = RuntimeFacts(
        steps=(
            RuntimeStep("import torch", True),
            RuntimeStep("cuda available", False, error=detail),
        ),
        failure=ProbeFailure("runtime", FailureKind.NONZERO_EXIT, detail),
    )
    [result] = results(replace(ENV, runtime=crashed), "GSI-TORCH-003")
    assert result.summary == "The interpreter did not survive using CUDA"
    assert result.reason == detail


def test_a_failing_import_fails() -> None:
    broken = RuntimeFacts(
        steps=(RuntimeStep("import torch", False, error="ImportError: libcudnn.so.9"),)
    )
    [result] = results(replace(ENV, runtime=broken), "GSI-TORCH-003")
    assert (result.summary, result.reason) == ("import torch fails", "ImportError: libcudnn.so.9")
