"""Rules: pure functions from a snapshot to findings, registered under stable codes."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from gpu_stack_inspector.models import FINDING_ID, Finding, Snapshot


@dataclass(frozen=True, slots=True)
class RuleContext:
    """What the user asked for, which changes how some facts are judged.

    ``cuda_expected``: the user wants these environments to use an NVIDIA GPU (the
    reason to run this tool), so a missing driver is a problem rather than a fact.
    ``shell_python``: the interpreter the user's shell runs, when known.
    """

    cuda_expected: bool = True
    shell_python: str | None = None


RuleCheck = Callable[[Snapshot, RuleContext], Iterable[Finding]]


@dataclass(frozen=True, slots=True)
class Rule:
    id: str
    title: str
    explanation: str
    check: RuleCheck
    sources: tuple[str, ...] = ()


class Registry:
    """Rules in registration order, which is the order they are evaluated and listed."""

    def __init__(self) -> None:
        self._rules: dict[str, Rule] = {}

    def rule(
        self, rule_id: str, title: str, *, sources: Iterable[str] = ()
    ) -> Callable[[RuleCheck], RuleCheck]:
        """Register a check; its docstring becomes the explanation shown by `explain`."""

        def register(check: RuleCheck) -> RuleCheck:
            if not FINDING_ID.fullmatch(rule_id):
                raise ValueError(f"rule id {rule_id!r} does not match GSI-<AREA>-<NNN>")
            if rule_id in self._rules:
                raise ValueError(f"rule {rule_id} is registered twice")
            explanation = inspect.cleandoc(check.__doc__ or "")
            if not explanation:
                raise ValueError(f"rule {rule_id} has no docstring to explain it")
            self._rules[rule_id] = Rule(rule_id, title, explanation, check, tuple(sources))
            return check

        return register

    def __iter__(self) -> Iterator[Rule]:
        return iter(self._rules.values())

    def get(self, rule_id: str) -> Rule | None:
        return self._rules.get(rule_id.upper())
