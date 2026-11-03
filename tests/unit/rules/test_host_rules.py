"""Host checks on hand-made snapshots."""

from __future__ import annotations

from gpu_stack_inspector.models import (
    FailureKind,
    Finding,
    GpuDevice,
    NvidiaFacts,
    ProbeFailure,
    Snapshot,
    Status,
)
from gpu_stack_inspector.rules.catalog import builtin_rules
from gpu_stack_inspector.rules.engine import RuleContext, evaluate

WORKING = NvidiaFacts(
    "/usr/bin/nvidia-smi",
    "595.99.02",
    "13.2",
    (GpuDevice(0, "NVIDIA GeForce RTX 3050 Laptop GPU", 4096, "8.6"),),
)


def findings(
    snapshot: Snapshot, context: RuleContext | None = None, prefix: str = "GSI-SYS"
) -> dict[str, Finding]:
    found = evaluate(snapshot, builtin_rules(), context)
    return {f.id: f for f in found if f.id.startswith(prefix)}


def snapshot(**fields: object) -> Snapshot:
    return Snapshot("0.1.0", "t", **fields)  # type: ignore[arg-type]


def test_nvidia_smi_found_passes() -> None:
    result = findings(snapshot(nvidia=WORKING))["GSI-SYS-001"]
    assert result.status is Status.PASS
    assert "/usr/bin/nvidia-smi" in result.reason


def test_missing_nvidia_smi_is_a_warning_when_cuda_is_expected() -> None:
    result = findings(snapshot(nvidia=NvidiaFacts(None)))["GSI-SYS-001"]
    assert result.status is Status.WARN
    assert result.verify_command == "nvidia-smi"
    assert result.limitations


def test_missing_nvidia_smi_does_not_apply_when_cuda_is_not_expected() -> None:
    result = findings(snapshot(nvidia=NvidiaFacts(None)), RuleContext(cuda_expected=False))
    assert result["GSI-SYS-001"].status is Status.NOT_APPLICABLE


def test_host_checks_are_skipped_when_the_host_was_not_inspected() -> None:
    assert findings(snapshot()) == {}


def test_a_working_driver_passes_and_explains_its_cuda_version() -> None:
    result = findings(snapshot(nvidia=WORKING))["GSI-SYS-002"]
    assert result.status is Status.PASS
    assert result.summary == "NVIDIA driver 595.99.02 answers"
    assert "supports CUDA up to 13.2" in result.reason
    assert "not a CUDA toolkit" in result.reason


def test_without_nvidia_smi_the_driver_is_not_tested_because_of_sys_001() -> None:
    result = findings(snapshot(nvidia=NvidiaFacts(None)))["GSI-SYS-002"]
    assert result.status is Status.NOT_TESTED
    assert result.caused_by == "GSI-SYS-001"


def test_a_driver_that_does_not_answer_fails_with_nvidia_smis_message() -> None:
    message = (
        "nvidia-smi exited with code 9: NVIDIA-SMI has failed because it couldn't "
        "communicate with the NVIDIA driver."
    )
    dead = NvidiaFacts(
        "/usr/bin/nvidia-smi",
        failures=(ProbeFailure("nvidia-smi", FailureKind.NONZERO_EXIT, message),),
    )
    result = findings(snapshot(nvidia=dead))["GSI-SYS-002"]
    assert result.status is Status.FAIL
    assert result.reason == message
    assert "lsmod" in (result.recommended_action or "")


def test_partial_problems_with_a_working_driver_become_limitations() -> None:
    partly = NvidiaFacts(
        "/usr/bin/nvidia-smi",
        "470.256.02",
        None,
        (GpuDevice(0, "Tesla K80"),),
        (ProbeFailure("nvidia-smi", FailureKind.MALFORMED_OUTPUT, "no maximum CUDA version"),),
    )
    result = findings(snapshot(nvidia=partly))["GSI-SYS-002"]
    assert result.status is Status.PASS
    assert "an unknown version" in result.reason
    assert result.limitations == ("no maximum CUDA version",)
