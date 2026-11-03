"""Host checks on hand-made snapshots."""

from __future__ import annotations

from gpu_stack_inspector.models import (
    Finding,
    GpuDevice,
    NvidiaFacts,
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
