"""Host checks on hand-made snapshots."""

from __future__ import annotations

import pytest

from gpu_stack_inspector.models import (
    EnvironmentVariable,
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


def test_a_visible_gpu_passes_with_its_name_and_compute_capability() -> None:
    result = findings(snapshot(nvidia=WORKING))["GSI-SYS-003"]
    assert result.status is Status.PASS
    assert result.summary == "1 GPU visible to the driver"
    assert result.reason == "NVIDIA GeForce RTX 3050 Laptop GPU (compute 8.6)"


def test_no_gpu_with_a_working_driver_fails() -> None:
    no_gpu = NvidiaFacts("/usr/bin/nvidia-smi", "595.99.02", "13.2")
    result = findings(snapshot(nvidia=no_gpu))["GSI-SYS-003"]
    assert result.status is Status.FAIL
    assert result.verify_command == "nvidia-smi -L"


def test_gpus_are_not_tested_when_the_driver_could_not_be_asked() -> None:
    assert findings(snapshot(nvidia=NvidiaFacts(None)))["GSI-SYS-003"].caused_by == "GSI-SYS-001"
    dead = NvidiaFacts("/usr/bin/nvidia-smi")
    assert findings(snapshot(nvidia=dead))["GSI-SYS-003"].caused_by == "GSI-SYS-002"


def test_one_missing_tool_gives_one_warning_and_two_consequences() -> None:
    result = findings(snapshot(nvidia=NvidiaFacts(None)))
    assert [(f.id, f.status, f.caused_by) for f in result.values()] == [
        ("GSI-SYS-001", Status.WARN, None),
        ("GSI-SYS-002", Status.NOT_TESTED, "GSI-SYS-001"),
        ("GSI-SYS-003", Status.NOT_TESTED, "GSI-SYS-001"),
    ]


def env_finding(value: str, nvidia: NvidiaFacts = WORKING) -> Finding:
    snap = snapshot(
        nvidia=nvidia, cuda_environment=(EnvironmentVariable("CUDA_VISIBLE_DEVICES", value),)
    )
    return findings(snap, prefix="GSI-ENV")["GSI-ENV-003"]


@pytest.mark.parametrize("value", ["", "-1", " ", "NoDevFiles"])
def test_values_that_hide_every_gpu_warn(value: str) -> None:
    result = env_finding(value)
    assert result.status is Status.WARN
    assert result.summary == "CUDA_VISIBLE_DEVICES hides every GPU"


def test_an_index_beyond_the_gpus_stops_cuda_there() -> None:
    result = env_finding("1,0")
    assert result.status is Status.WARN
    assert result.summary == "CUDA_VISIBLE_DEVICES names GPU 1, which does not exist"
    assert "no GPU is visible" in result.reason


def test_devices_before_an_invalid_index_stay_visible() -> None:
    two = NvidiaFacts(
        "/usr/bin/nvidia-smi", "595.99.02", "13.2", (GpuDevice(0, "A"), GpuDevice(1, "B"))
    )
    result = env_finding("0,5,1", two)
    assert "only GPU(s) 0 are visible" in result.reason


def test_a_valid_selection_passes() -> None:
    result = env_finding("0")
    assert result.status is Status.PASS
    assert result.limitations == ()


def test_uuids_pass_with_a_limitation() -> None:
    result = env_finding("GPU-a3f5e3f2")
    assert result.status is Status.PASS
    assert result.confidence.value == "medium"
    assert result.limitations


def test_nothing_is_said_when_the_variable_is_not_set() -> None:
    assert "GSI-ENV-003" not in findings(snapshot(nvidia=WORKING), prefix="GSI-ENV")
