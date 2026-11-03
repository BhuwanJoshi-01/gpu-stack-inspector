from gpu_stack_inspector.models import Confidence, Severity, Status


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
