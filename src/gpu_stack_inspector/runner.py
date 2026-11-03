"""Bounded, shell-free execution of diagnostic commands.

Every external command the tool runs goes through a :class:`CommandRunner`, so that
collectors can be tested with a fake one and the real one can enforce the same limits
everywhere: an argument list (never a shell), a timeout, an output cap and no stdin.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

DEFAULT_MAX_OUTPUT = 256 * 1024
"""Bytes kept per stream; diagnostic output larger than this is truncated."""


class Outcome(StrEnum):
    """How a command ended, as far as the runner can tell."""

    OK = "ok"
    NONZERO = "nonzero"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    PERMISSION = "permission"
    OS_ERROR = "os_error"


@dataclass(frozen=True, slots=True)
class CommandResult:
    """What a command printed and how it ended.

    ``returncode`` is ``None`` when the command never started or was killed after a
    timeout. ``error`` holds the operating system's message when it never started.
    """

    argv: tuple[str, ...]
    outcome: Outcome
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    duration: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome is Outcome.OK


class CommandRunner(Protocol):
    """Runs one command and reports the result; never raises for command failures."""

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        env_overrides: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> CommandResult: ...
