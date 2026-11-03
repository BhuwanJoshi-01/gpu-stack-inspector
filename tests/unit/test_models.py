import dataclasses

import pytest

from gpu_stack_inspector.models import Confidence, Evidence, Finding, Severity, Status


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
