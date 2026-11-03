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
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import IO, Protocol

DEFAULT_MAX_OUTPUT = 256 * 1024
"""Bytes kept per stream; diagnostic output larger than this is truncated."""

_REAP_TIMEOUT = 2.0
"""Seconds to wait for a finished command's pipes to close before giving up on them."""

_CHUNK = 64 * 1024


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

        if proc.stdout is None or proc.stderr is None:  # cannot happen with PIPE
            raise RuntimeError("subprocess pipes were not created")
        readers = (
            _CappedReader(proc.stdout, self.max_output),
            _CappedReader(proc.stderr, self.max_output),
        )
        for reader in readers:
            reader.start()
        timed_out = False
        try:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill(proc)
                proc.wait()
            # A child that escaped the process group may hold a pipe open; don't wait
            # for it forever, keep what was read so far.
            deadline = time.monotonic() + _REAP_TIMEOUT
            for reader in readers:
                reader.join(max(0.0, deadline - time.monotonic()))
        except BaseException:
            # Ctrl+C or an internal error: never leave the command running behind us.
            _kill(proc)
            raise

        out, err = readers
        return CommandResult(
            argv=args,
            outcome=Outcome.TIMEOUT
            if timed_out
            else (Outcome.OK if proc.returncode == 0 else Outcome.NONZERO),
            returncode=None if timed_out else proc.returncode,
            stdout=_decode(out.data),
            stderr=_decode(err.data),
            stdout_truncated=out.truncated,
            stderr_truncated=err.truncated,
            duration=time.monotonic() - started,
        )


class _CappedReader(threading.Thread):
    """Drains one pipe, keeping at most ``limit`` bytes.

    It keeps reading (and discarding) past the limit so the command never blocks on a
    full pipe.
    """

    def __init__(self, stream: IO[bytes], limit: int) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._limit = limit
        self._kept = bytearray()
        self.truncated = False

    def run(self) -> None:
        # os.read returns as soon as any bytes are available; a buffered read() would
        # wait for a full chunk and lose the output of a command whose child keeps the
        # pipe open.
        fd = self._stream.fileno()
        with contextlib.suppress(OSError, ValueError), self._stream:
            while chunk := os.read(fd, _CHUNK):
                room = self._limit - len(self._kept)
                if room > 0:
                    self._kept += chunk[:room]
                if len(chunk) > room:
                    self.truncated = True

    @property
    def data(self) -> bytes:
        return bytes(self._kept)


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


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")
