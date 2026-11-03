"""Checks of each inspected interpreter: usable, the shell's own, its PyTorch."""

from __future__ import annotations

from collections.abc import Iterator

from gpu_stack_inspector.models import (
    Confidence,
    Evidence,
    FailureKind,
    Finding,
    InterpreterFacts,
    Severity,
    Snapshot,
    Status,
)
from gpu_stack_inspector.rules.catalog import RULES
from gpu_stack_inspector.rules.engine import RuleContext

_CANNOT_START = frozenset({FailureKind.NOT_FOUND, FailureKind.PERMISSION_DENIED})


def _usable(interpreter: InterpreterFacts) -> bool:
    return interpreter.version is not None


@RULES.rule("GSI-ENV-001", "The interpreter can be inspected")
def interpreter_usable(snapshot: Snapshot, context: RuleContext) -> Iterator[Finding]:
    """Whether each named interpreter started and answered the metadata probe.

    "Cannot be started" (a wrong path, a deleted environment, no execute permission) is
    told apart from "starts and crashes" (a broken environment, a damaged standard
    library), because they need different fixes. An interpreter that answered only in
    part passes with the problems listed as limitations.
    """
    for interpreter in snapshot.interpreters:
        details = tuple(f.detail for f in interpreter.failures)
        if _usable(interpreter):
            kind = ", ".join(
                part
                for part in (
                    interpreter.implementation,
                    "conda env" if interpreter.conda else None,
                    "venv" if interpreter.virtualenv else None,
                )
                if part
            )
            yield Finding(
                id="GSI-ENV-001",
                status=Status.PASS,
                severity=Severity.INFO,
                confidence=Confidence.HIGH if not details else Confidence.MEDIUM,
                summary=f"Python {interpreter.version} ({kind or 'python'})",
                reason=f"Inspected {interpreter.path}.",
                subject=interpreter.label,
                evidence=(Evidence("python-probe", "executable", interpreter.executable),),
                limitations=details,
            )
            continue
        cannot_start = any(f.kind in _CANNOT_START for f in interpreter.failures)
        yield Finding(
            id="GSI-ENV-001",
            status=Status.FAIL,
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            summary=(
                "The interpreter cannot be started"
                if cannot_start
                else "The interpreter started but did not answer"
            ),
            reason=details[0] if details else "No answer from the metadata probe.",
            subject=interpreter.label,
            evidence=tuple(Evidence("python-probe", "failure", detail) for detail in details),
            recommended_action=(
                "Check the path: the environment may have been moved or deleted."
                if cannot_start
                else "Run the interpreter by hand to see the whole error; a damaged "
                "environment usually has to be recreated."
            ),
            verify_command=f"{interpreter.path} --version",
        )
