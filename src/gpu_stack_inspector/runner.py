"""Bounded, shell-free execution of diagnostic commands.

Every external command the tool runs goes through a :class:`CommandRunner`, so that
collectors can be tested with a fake one and the real one can enforce the same limits
everywhere: an argument list (never a shell), a timeout, an output cap and no stdin.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

DEFAULT_MAX_OUTPUT = 256 * 1024
"""Bytes kept per stream; diagnostic output larger than this is truncated."""

_REAP_TIMEOUT = 2.0
"""Seconds to wait for a killed command's pipes to close before giving up on them."""


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


class SubprocessRunner:
    """The real runner: ``subprocess`` with an argument list, never a shell.

    Each command runs in its own session (on POSIX), so a timeout or Ctrl+C kills the
    command together with any children it started. Windows only kills the command itself.
    """

    def __init__(self, max_output: int = DEFAULT_MAX_OUTPUT) -> None:
        if max_output <= 0:
            raise ValueError("max_output must be positive")
        self.max_output = max_output

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        env_overrides: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> CommandResult:
        args = _checked_argv(argv)
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        env = {**os.environ, **env_overrides} if env_overrides else None
        started = time.monotonic()
        try:
            proc = subprocess.Popen(  # noqa: S603 - argument list without a shell
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=cwd,
                start_new_session=sys.platform != "win32",
            )
        except FileNotFoundError as exc:
            return _not_started(args, Outcome.NOT_FOUND, exc, started)
        except PermissionError as exc:
            return _not_started(args, Outcome.PERMISSION, exc, started)
        except OSError as exc:
            return _not_started(args, Outcome.OS_ERROR, exc, started)

        timed_out = False
        try:
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill(proc)
                out, err = _collect_after_kill(proc)
        except BaseException:
            # Ctrl+C or an internal error: never leave the command running behind us.
            _kill(proc)
            raise

        return CommandResult(
            argv=args,
            outcome=Outcome.TIMEOUT
            if timed_out
            else (Outcome.OK if proc.returncode == 0 else Outcome.NONZERO),
            returncode=None if timed_out else proc.returncode,
            stdout=_decode(out),
            stderr=_decode(err),
            duration=time.monotonic() - started,
        )


def _checked_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, str):
        raise TypeError("argv must be a sequence of str, not a single command string")
    args = tuple(argv)
    if not args:
        raise ValueError("argv is empty")
    for arg in args:
        if not isinstance(arg, str):
            raise TypeError(f"argv items must be str, got {type(arg).__name__}")
    return args


def _not_started(
    args: tuple[str, ...], outcome: Outcome, exc: OSError, started: float
) -> CommandResult:
    return CommandResult(
        argv=args,
        outcome=outcome,
        error=exc.strerror or str(exc),
        duration=time.monotonic() - started,
    )


def _kill(proc: subprocess.Popen[bytes]) -> None:
    if sys.platform == "win32":
        proc.kill()
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):  # already gone
        os.killpg(proc.pid, signal.SIGKILL)


def _collect_after_kill(proc: subprocess.Popen[bytes]) -> tuple[bytes, bytes]:
    try:
        return proc.communicate(timeout=_REAP_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        # A child escaped the process group and still holds the pipes; keep what we have.
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
        proc.wait()
        return _as_bytes(exc.stdout), _as_bytes(exc.stderr)


def _as_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    return value.encode() if isinstance(value, str) else value


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")
