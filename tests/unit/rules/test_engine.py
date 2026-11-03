from collections.abc import Iterable

import pytest

from gpu_stack_inspector.models import Confidence, Finding, Severity, Snapshot, Status
from gpu_stack_inspector.rules.engine import Registry, RuleContext, evaluate


def no_findings(snapshot: Snapshot, context: RuleContext) -> Iterable[Finding]:
    """Never finds anything."""
    return ()


def also_nothing(snapshot: Snapshot, context: RuleContext) -> Iterable[Finding]:
    """Also never finds anything."""
    return ()


def test_rules_keep_registration_order_and_their_docstring() -> None:
    registry = Registry()
    registry.rule("GSI-SYS-002", "Second", sources=["https://example.invalid/doc"])(no_findings)
    registry.rule("GSI-SYS-001", "First")(also_nothing)
    rules = list(registry)
    assert [r.id for r in rules] == ["GSI-SYS-002", "GSI-SYS-001"]
    assert rules[0].explanation == "Never finds anything."
    assert rules[0].sources == ("https://example.invalid/doc",)


def test_a_rule_without_a_docstring_is_rejected() -> None:
    with pytest.raises(ValueError, match="no docstring"):
        Registry().rule("GSI-SYS-001", "Undocumented")(lambda s, c: ())  # type: ignore[arg-type]


def test_duplicate_and_malformed_ids_are_rejected() -> None:
    registry = Registry()
    registry.rule("GSI-SYS-001", "One")(no_findings)
    with pytest.raises(ValueError, match="twice"):
        registry.rule("GSI-SYS-001", "Again")(no_findings)
    with pytest.raises(ValueError, match="does not match"):
        registry.rule("SYS-1", "Bad")(no_findings)


def test_rules_are_found_by_id_case_insensitively() -> None:
    registry = Registry()
    registry.rule("GSI-TORCH-002", "Build")(no_findings)
    rule = registry.get("gsi-torch-002")
    assert rule is not None
    assert rule.title == "Build"
    assert registry.get("GSI-TORCH-999") is None


def _finding(rule_id: str, subject: str = "host") -> Finding:
    return Finding(
        id=rule_id,
        status=Status.PASS,
        severity=Severity.INFO,
        confidence=Confidence.HIGH,
        summary="ok",
        reason="because",
        subject=subject,
    )


SNAPSHOT = Snapshot(tool_version="0.1.0", created_at="t")


def test_findings_come_out_in_rule_order() -> None:
    registry = Registry()

    @registry.rule("GSI-SYS-001", "Host")
    def host(snapshot: Snapshot, context: RuleContext) -> Iterable[Finding]:
        """Host check."""
        yield _finding("GSI-SYS-001")

    @registry.rule("GSI-ENV-001", "Interpreters")
    def interpreters(snapshot: Snapshot, context: RuleContext) -> Iterable[Finding]:
        """One finding per interpreter."""
        yield _finding("GSI-ENV-001", "A")
        yield _finding("GSI-ENV-001", "B")

    findings = evaluate(SNAPSHOT, registry)
    assert [(f.id, f.subject) for f in findings] == [
        ("GSI-SYS-001", "host"),
        ("GSI-ENV-001", "A"),
        ("GSI-ENV-001", "B"),
    ]


def test_the_context_reaches_the_rules() -> None:
    registry = Registry()
    seen: list[RuleContext] = []

    @registry.rule("GSI-SYS-001", "Host")
    def host(snapshot: Snapshot, context: RuleContext) -> Iterable[Finding]:
        """Records the context."""
        seen.append(context)
        return ()

    evaluate(SNAPSHOT, registry, RuleContext(cuda_expected=False))
    evaluate(SNAPSHOT, registry)
    assert [c.cuda_expected for c in seen] == [False, True]


def test_a_broken_rule_becomes_an_unknown_finding_and_the_others_still_run() -> None:
    registry = Registry()

    @registry.rule("GSI-SYS-001", "Driver tools")
    def broken(snapshot: Snapshot, context: RuleContext) -> Iterable[Finding]:
        """Raises."""
        raise KeyError("gpus")

    @registry.rule("GSI-SYS-002", "Driver")
    def fine(snapshot: Snapshot, context: RuleContext) -> Iterable[Finding]:
        """Works."""
        yield _finding("GSI-SYS-002")

    first, second = evaluate(SNAPSHOT, registry)
    assert (first.id, first.status) == ("GSI-SYS-001", Status.UNKNOWN)
    assert "internal error" in first.summary
    assert "KeyError" in first.reason
    assert (second.id, second.status) == ("GSI-SYS-002", Status.PASS)


def test_a_rule_reporting_under_another_code_is_a_bug() -> None:
    registry = Registry()

    @registry.rule("GSI-SYS-001", "Host")
    def wrong(snapshot: Snapshot, context: RuleContext) -> Iterable[Finding]:
        """Reports under the wrong code."""
        yield _finding("GSI-SYS-002")

    with pytest.raises(ValueError, match="reported a finding as GSI-SYS-002"):
        evaluate(SNAPSHOT, registry)
