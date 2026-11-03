import dataclasses
import re

import pytest

from gpu_stack_inspector.models import (
    SCHEMA_VERSION,
    Confidence,
    Evidence,
    FailureKind,
    Finding,
    GpuDevice,
    InterpreterFacts,
    NvidiaFacts,
    ProbeFailure,
    Severity,
    Snapshot,
    Status,
    utc_timestamp,
)


def test_status_values_are_the_documented_json_strings() -> None:
    assert [s.value for s in Status] == [
        "pass",
        "warn",
        "fail",
        "unknown",
        "not_applicable",
        "not_tested",
    ]


def test_enums_compare_equal_to_their_json_strings() -> None:
    assert Status.UNKNOWN == "unknown"
    assert Severity.HIGH == "high"
    assert Confidence.MEDIUM == "medium"


def _finding(**overrides: object) -> Finding:
    values: dict[str, object] = {
        "id": "GSI-TORCH-002",
        "status": Status.FAIL,
        "severity": Severity.HIGH,
        "confidence": Confidence.HIGH,
        "summary": "The PyTorch build has no CUDA support.",
        "reason": "The wheel is a CPU-only build.",
    }
    values.update(overrides)
    return Finding(**values)  # type: ignore[arg-type]


def test_finding_defaults_to_host_subject_and_no_evidence() -> None:
    finding = _finding()
    assert finding.subject == "host"
    assert finding.evidence == ()
    assert finding.limitations == ()


@pytest.mark.parametrize(
    "bad_id", ["GSI-TORCH-2", "gsi-torch-002", "GPU-TORCH-002", "GSI-TORCH-0021", ""]
)
def test_finding_rejects_malformed_ids(bad_id: str) -> None:
    with pytest.raises(ValueError, match="does not match"):
        _finding(id=bad_id)


def test_finding_rejects_empty_summary() -> None:
    with pytest.raises(ValueError, match="empty summary"):
        _finding(summary="  ")


def test_findings_and_evidence_are_immutable() -> None:
    finding = _finding(evidence=(Evidence("nvidia-smi", "driver_version", "595.99"),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        finding.status = Status.PASS  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        finding.evidence[0].value = "1"  # type: ignore[misc]


def test_failure_kinds_are_the_documented_json_strings() -> None:
    assert [k.value for k in FailureKind] == [
        "not_found",
        "permission_denied",
        "timeout",
        "nonzero_exit",
        "malformed_output",
        "os_error",
        "probe_error",
    ]


def test_probe_failure_keeps_probe_kind_and_detail() -> None:
    failure = ProbeFailure("nvidia-smi", FailureKind.TIMEOUT, "no answer within 8 s")
    assert (failure.probe, failure.kind, failure.detail) == (
        "nvidia-smi",
        FailureKind.TIMEOUT,
        "no answer within 8 s",
    )


def test_snapshot_carries_schema_and_tool_version() -> None:
    snapshot = Snapshot(tool_version="0.1.0", created_at="2026-11-03T10:00:00Z")
    assert snapshot.schema_version == SCHEMA_VERSION == 1
    assert snapshot.findings == ()
    assert snapshot.probe_failures == ()


def test_utc_timestamp_is_iso_8601_utc() -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", utc_timestamp())


def test_nvidia_facts_without_nvidia_smi_have_no_gpus_and_no_failures() -> None:
    facts = NvidiaFacts(smi_path=None)
    assert facts.gpus == ()
    assert facts.failures == ()
    assert facts.max_cuda_version is None


def test_gpu_device_values_default_to_unknown() -> None:
    gpu = GpuDevice(index=0, name="NVIDIA GeForce RTX 3050 Laptop GPU")
    assert gpu.memory_total_mib is None
    assert gpu.compute_capability is None


def test_virtualenv_is_known_only_when_both_prefixes_are() -> None:
    base = InterpreterFacts(label="A", path="/envs/a/bin/python")
    assert base.virtualenv is None
    venv = InterpreterFacts(label="A", path="/p/bin/python", prefix="/p", base_prefix="/usr")
    assert venv.virtualenv is True
    plain = InterpreterFacts(label="B", path="/usr/bin/python3", prefix="/usr", base_prefix="/usr")
    assert plain.virtualenv is False
