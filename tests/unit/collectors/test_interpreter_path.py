"""Validating the interpreter paths a user passes with --python."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_stack_inspector.models import FailureKind

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX permissions and symlinks")


def check(raw: str, cwd: Path, **kwargs: Any) -> Any:
    from gpu_stack_inspector.collectors.python_env import interpreter_path

    return interpreter_path(raw, cwd=cwd, **kwargs)


def fake_python(directory: Path, name: str = "python") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    program = directory / name
    program.write_text("#!/bin/sh\n")
    program.chmod(0o755)
    return program


def test_an_absolute_executable_path_is_accepted(tmp_path: Path) -> None:
    program = fake_python(tmp_path / "env" / "bin")
    path, failure = check(str(program), tmp_path)
    assert (path, failure) == (program, None)


def test_a_relative_path_is_made_absolute_against_the_working_directory(tmp_path: Path) -> None:
    program = fake_python(tmp_path / ".venv" / "bin")
    path, failure = check(str(Path(".venv", "bin", "python")), tmp_path)
    assert failure is None
    assert path == tmp_path / ".venv" / "bin" / "python"
    assert path == program


def test_home_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    program = fake_python(tmp_path / "envs" / "a" / "bin")
    path, failure = check("~/envs/a/bin/python", Path("/"))
    assert (path, failure) == (program, None)


@posix_only
def test_symlinks_are_kept_so_a_venv_python_stays_in_its_venv(tmp_path: Path) -> None:
    link = tmp_path / "venv" / "bin" / "python"
    link.parent.mkdir(parents=True)
    link.symlink_to(sys.executable)
    path, failure = check(str(link), tmp_path)
    assert failure is None
    assert path == link
    assert path != Path(sys.executable).resolve()


def test_a_missing_path_is_not_found(tmp_path: Path) -> None:
    path, failure = check(str(tmp_path / "nope" / "python"), tmp_path)
    assert path is None
    assert failure.kind is FailureKind.NOT_FOUND
    assert failure.probe == "python"
    assert "does not exist" in failure.detail


@posix_only
def test_a_dangling_symlink_says_where_it_points(tmp_path: Path) -> None:
    link = tmp_path / "python"
    link.symlink_to(tmp_path / "deleted-env" / "bin" / "python3.12")
    path, failure = check(str(link), tmp_path)
    assert path is None
    assert failure.kind is FailureKind.NOT_FOUND
    assert "deleted-env" in failure.detail


@posix_only
def test_a_file_without_execute_permission_is_rejected(tmp_path: Path) -> None:
    program = fake_python(tmp_path)
    program.chmod(0o644)
    path, failure = check(str(program), tmp_path)
    assert path is None
    assert failure.kind is FailureKind.PERMISSION_DENIED


def test_an_empty_path_is_rejected(tmp_path: Path) -> None:
    path, failure = check("  ", tmp_path)
    assert path is None
    assert failure.kind is FailureKind.NOT_FOUND


def test_an_environment_directory_suggests_its_interpreter(tmp_path: Path) -> None:
    program = fake_python(tmp_path / "envs" / "a" / "bin")
    path, failure = check(str(tmp_path / "envs" / "a"), tmp_path)
    assert path is None
    assert failure.kind is FailureKind.NOT_FOUND
    assert failure.detail.endswith(f"did you mean {program}?")


def test_a_plain_directory_gets_no_suggestion(tmp_path: Path) -> None:
    path, failure = check(str(tmp_path), tmp_path)
    assert path is None
    assert failure.detail == f"{tmp_path} is a directory, not an interpreter"


def test_on_windows_a_path_without_exe_finds_the_exe(tmp_path: Path) -> None:
    program = fake_python(tmp_path / "venv" / "Scripts", "python.exe")
    path, failure = check(str(tmp_path / "venv" / "Scripts" / "python"), tmp_path, windows=True)
    assert (path, failure) == (program, None)


def test_elsewhere_a_missing_path_is_not_completed_with_exe(tmp_path: Path) -> None:
    fake_python(tmp_path / "venv" / "Scripts", "python.exe")
    path, failure = check(str(tmp_path / "venv" / "Scripts" / "python"), tmp_path, windows=False)
    assert path is None
    assert failure.kind is FailureKind.NOT_FOUND
