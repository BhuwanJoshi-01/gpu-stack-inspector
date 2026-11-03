"""Running the runtime probe and reading its answer (scripted runner)."""

from __future__ import annotations

import tempfile
from typing import Any

import pytest

from gpu_stack_inspector.models import FailureKind, GpuDevice
from gpu_stack_inspector.runner import CommandResult, Outcome
from tests.fakes import FakeRunner, read_fixture

pending = pytest.mark.xfail(strict=True, reason="GPU-008: runtime probe runner pending")
PYTHON = "/envs/a/bin/python"


def module() -> Any:
    from gpu_stack_inspector.probes import runtime_probe

    return runtime_probe


def run(**answer: Any) -> tuple[Any, FakeRunner]:
    argv = (PYTHON, "-c", module().program_source())
    runner = FakeRunner()
    if "result" in answer:
        runner.script_result(CommandResult(argv, **answer["result"]))
    else:
        runner.script(*argv, **answer)
    return module().run_runtime_probe(runner, PYTHON), runner


@pending
def test_a_working_gpu_environment() -> None:
    facts, runner = run(stdout=read_fixture("runtime", "sapiens2.stdout.txt"))
    assert (facts.torch_version, facts.cuda_version) == ("2.11.0+cu128", "12.8")
    assert facts.cuda_available is True
    assert facts.devices == (GpuDevice(0, "NVIDIA GeForce RTX 3050 Laptop GPU", None, "8.6"),)
    assert facts.tensor_ok is True
    assert [s.step for s in facts.steps] == [
        "import torch",
        "cuda available",
        "devices",
        "tensor operation",
    ]
    assert all(s.ok for s in facts.steps)
    assert facts.failure is None
    call = runner.calls[0]
    assert call.timeout == 60.0
    assert call.cwd is not None
    assert str(call.cwd).startswith(tempfile.gettempdir())


@pending
def test_the_runtime_version_can_carry_a_tag_the_metadata_lacks() -> None:
    facts, _ = run(stdout=read_fixture("runtime", "wilor.stdout.txt"))
    assert facts.torch_version == "2.14.1+cu130"
    assert facts.tensor_ok is True


@pending
def test_a_cpu_only_build_is_not_available_and_not_exercised() -> None:
    facts, _ = run(stdout=read_fixture("runtime", "cpu-venv.stdout.txt"))
    assert (facts.cuda_available, facts.tensor_ok, facts.cuda_version) == (False, None, None)
    assert facts.failure is None


@pending
@pytest.mark.parametrize("returncode", [-11, 139])
def test_a_segfault_names_the_step_it_happened_in(returncode: int) -> None:
    stdout = "GSI-RUNTIME-STEP import torch\nGSI-RUNTIME-STEP cuda available\n"
    facts, _ = run(result={"outcome": Outcome.NONZERO, "returncode": returncode, "stdout": stdout})
    assert [(s.step, s.ok) for s in facts.steps] == [
        ("import torch", True),
        ("cuda available", False),
    ]
    assert facts.failure.kind is FailureKind.NONZERO_EXIT
    assert "segmentation fault (SIGSEGV)" in facts.failure.detail
    assert "'cuda available'" in facts.failure.detail
    assert facts.cuda_available is None


@pending
def test_an_abort_is_named() -> None:
    stdout = "GSI-RUNTIME-STEP import torch\n"
    facts, _ = run(
        result={
            "outcome": Outcome.NONZERO,
            "returncode": -6,
            "stdout": stdout,
            "stderr": "terminate called",
        }
    )
    assert "abort (SIGABRT)" in facts.failure.detail


@pending
def test_a_hang_is_a_timeout_in_the_step_it_was_in() -> None:
    stdout = (
        "GSI-RUNTIME-STEP import torch\nGSI-RUNTIME-STEP cuda available\nGSI-RUNTIME-STEP devices\n"
    )
    facts, _ = run(result={"outcome": Outcome.TIMEOUT, "stdout": stdout, "duration": 60.0})
    assert facts.failure.kind is FailureKind.TIMEOUT
    assert "'devices'" in facts.failure.detail
    assert [(s.step, s.ok) for s in facts.steps][-1] == ("devices", False)


@pending
def test_a_python_error_outside_the_steps_is_explained() -> None:
    stderr = (
        "Traceback (most recent call last):\nModuleNotFoundError: No module named 'encodings'\n"
    )
    facts, _ = run(result={"outcome": Outcome.NONZERO, "returncode": 1, "stderr": stderr})
    assert facts.failure.detail.endswith("ModuleNotFoundError: No module named 'encodings'")
    assert facts.steps == ()


@pending
def test_no_result_is_malformed_output() -> None:
    facts, _ = run(stdout="hello\n")
    assert facts.failure.kind is FailureKind.MALFORMED_OUTPUT
