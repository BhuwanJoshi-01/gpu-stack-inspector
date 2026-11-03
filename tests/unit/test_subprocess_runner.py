"""The real runner, exercised with short-lived child Python processes."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from gpu_stack_inspector.runner import CommandResult, Outcome

PY = sys.executable
pending = pytest.mark.xfail(strict=True, reason="GPU-002: SubprocessRunner pending")
posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX permissions and process groups")


def run(argv: list[str], *, timeout: float = 10.0, **kwargs: object) -> CommandResult:
    from gpu_stack_inspector.runner import SubprocessRunner

    max_output = kwargs.pop("max_output", None)
    runner = SubprocessRunner() if max_output is None else SubprocessRunner(max_output=max_output)  # type: ignore[arg-type]
    return runner.run(argv, timeout=timeout, **kwargs)  # type: ignore[arg-type]


@pending
def test_successful_command_is_ok_with_its_output() -> None:
    result = run([PY, "-c", "print('hello')"])
    assert result.outcome is Outcome.OK
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"
    assert result.argv == (PY, "-c", "print('hello')")
    assert result.duration >= 0


@pending
def test_nonzero_exit_keeps_returncode_and_stderr() -> None:
    result = run([PY, "-c", "import sys; sys.stderr.write('broken'); sys.exit(3)"])
    assert result.outcome is Outcome.NONZERO
    assert result.returncode == 3
    assert result.stderr == "broken"


@pending
def test_missing_program_is_not_found() -> None:
    result = run(["gsi-test-no-such-program-0f3a"])
    assert result.outcome is Outcome.NOT_FOUND
    assert result.returncode is None
    assert result.error


@pending
@posix_only
def test_file_without_execute_permission_is_permission(tmp_path: Path) -> None:
    script = tmp_path / "not-executable"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o644)
    result = run([str(script)])
    assert result.outcome is Outcome.PERMISSION


@pending
def test_arguments_are_never_interpreted_by_a_shell() -> None:
    payload = "$HOME; echo injected `id`"
    result = run([PY, "-c", "import sys; print(sys.argv[1])", payload])
    assert result.stdout.strip() == payload


@pending
def test_stdin_is_closed_so_reading_it_does_not_hang() -> None:
    result = run([PY, "-c", "import sys; print(repr(sys.stdin.read()))"], timeout=5)
    assert result.outcome is Outcome.OK
    assert result.stdout.strip() == "''"


@pending
def test_env_overrides_are_added_to_the_inherited_environment() -> None:
    code = "import os; print(os.environ['GSI_TEST_VALUE'], 'PATH' in os.environ)"
    result = run([PY, "-c", code], env_overrides={"GSI_TEST_VALUE": "42"})
    assert result.stdout.split() == ["42", "True"]


@pending
def test_cwd_is_used(tmp_path: Path) -> None:
    result = run([PY, "-c", "import os; print(os.getcwd())"], cwd=tmp_path)
    assert Path(result.stdout.strip()).resolve() == tmp_path.resolve()


@pending
def test_undecodable_bytes_are_replaced_not_fatal() -> None:
    result = run([PY, "-c", "import sys; sys.stdout.buffer.write(b'ok \\xff\\xfe end')"])
    assert result.outcome is Outcome.OK
    assert result.stdout == "ok �� end"


@pending
def test_empty_argv_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        run([])


@pending
def test_non_string_arguments_are_rejected() -> None:
    with pytest.raises(TypeError, match="str"):
        run([PY, "-c", 42])  # type: ignore[list-item]


@pending
def test_output_beyond_the_cap_is_truncated_on_both_streams() -> None:
    code = "import sys; sys.stdout.write('o' * 50000); sys.stderr.write('e' * 50000)"
    result = run([PY, "-c", code], max_output=1000)
    assert result.outcome is Outcome.OK
    assert result.stdout == "o" * 1000
    assert result.stderr == "e" * 1000
    assert result.stdout_truncated
    assert result.stderr_truncated


@pending
def test_output_within_the_cap_is_not_marked_truncated() -> None:
    result = run([PY, "-c", "print('short')"], max_output=1000)
    assert not result.stdout_truncated
    assert not result.stderr_truncated


@pending
def test_timeout_is_reported_without_waiting_for_the_command() -> None:
    started = time.monotonic()
    result = run(
        [PY, "-c", "import time; print('started', flush=True); time.sleep(30)"], timeout=0.5
    )
    assert result.outcome is Outcome.TIMEOUT
    assert result.returncode is None
    assert time.monotonic() - started < 10
    assert "started" in result.stdout


@pending
@posix_only
def test_timeout_also_kills_the_commands_children(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    code = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    result = run([PY, "-c", code], timeout=1.0)
    assert result.outcome is Outcome.TIMEOUT
    child_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail(f"child process {child_pid} survived the timeout")
