from collections.abc import Iterable

import pytest

from gpu_stack_inspector.models import Finding, Snapshot
from gpu_stack_inspector.rules.engine import Registry, RuleContext


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
