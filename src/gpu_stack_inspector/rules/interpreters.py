"""Checks of each inspected interpreter: usable, the shell's own, its PyTorch."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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


def _environment_directory(path: str) -> str:
    # Same directory means same environment: bin/python and bin/python3 of one venv match,
    # two venvs do not. Symlinks are not followed, they would make every venv equal to
    # its base interpreter, which is why this is absolute() and not resolve().
    return str(Path(path).absolute().parent)


@RULES.rule("GSI-ENV-002", "The interpreter is the one the shell runs")
def shell_interpreter(snapshot: Snapshot, context: RuleContext) -> Iterator[Finding]:
    """Whether an inspected interpreter is the python the user's shell would start.

    A common source of confusion: the environment that was checked (or fixed) is not the
    one a plain `python` command runs, for example because another venv or conda env is
    active. Interpreters count as the same when they live in the same directory.
    """
    if context.shell_python is None:
        return
    shell_directory = _environment_directory(context.shell_python)
    for interpreter in snapshot.interpreters:
        if not _usable(interpreter):
            continue
        if _environment_directory(interpreter.path) == shell_directory:
            continue
        yield Finding(
            id="GSI-ENV-002",
            status=Status.WARN,
            severity=Severity.LOW,
            confidence=Confidence.MEDIUM,
            summary="This is not the python your shell runs",
            reason=(
                f"Your shell runs {context.shell_python}. Results for {interpreter.path} do not "
                "apply to programs started with a plain `python` command."
            ),
            subject=interpreter.label,
            evidence=(Evidence("PATH", "python", context.shell_python),),
            recommended_action=(
                "Activate this environment before running your program, or start it with "
                "this interpreter's full path."
            ),
            verify_command="which python",
            limitations=(
                "Interpreters are compared by directory, so two interpreters of one "
                "environment in different directories are reported as different.",
            ),
        )
