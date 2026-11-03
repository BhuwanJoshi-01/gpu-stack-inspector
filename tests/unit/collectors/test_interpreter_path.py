"""Validating the interpreter paths a user passes with --python."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_stack_inspector.models import FailureKind

pending = pytest.mark.xfail(strict=True, reason="GPU-003: interpreter path validation pending")
posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX permissions and symlinks")


def check(raw: str, cwd: Path) -> Any:
    from gpu_stack_inspector.collectors.python_env import interpreter_path

    return interpreter_path(raw, cwd=cwd)


def fake_python(directory: Path, name: str = "python") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    program = directory / name
    program.write_text("#!/bin/sh\n")
    program.chmod(0o755)
    return program


@pending
def test_an_absolute_executable_path_is_accepted(tmp_path: Path) -> None:
    program = fake_python(tmp_path / "env" / "bin")
    path, failure = check(str(program), tmp_path)
    assert (path, failure) == (program, None)


@pending
def test_a_relative_path_is_made_absolute_against_the_working_directory(tmp_path: Path) -> None:
    program = fake_python(tmp_path / ".venv" / "bin")
    path, failure = check(str(Path(".venv", "bin", "python")), tmp_path)
    assert failure is None
    assert path == tmp_path / ".venv" / "bin" / "python"
    assert path == program


@pending
def test_home_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    program = fake_python(tmp_path / "envs" / "a" / "bin")
    path, failure = check("~/envs/a/bin/python", Path("/"))
    assert (path, failure) == (program, None)


@pending
@posix_only
def test_symlinks_are_kept_so_a_venv_python_stays_in_its_venv(tmp_path: Path) -> None:
    link = tmp_path / "venv" / "bin" / "python"
    link.parent.mkdir(parents=True)
    link.symlink_to(sys.executable)
    path, failure = check(str(link), tmp_path)
    assert failure is None
    assert path == link
    assert path != Path(sys.executable).resolve()


@pending
def test_a_missing_path_is_not_found(tmp_path: Path) -> None:
    path, failure = check(str(tmp_path / "nope" / "python"), tmp_path)
    assert path is None
    assert failure.kind is FailureKind.NOT_FOUND
    assert failure.probe == "python"
    assert "does not exist" in failure.detail


@pending
@posix_only
def test_a_dangling_symlink_says_where_it_points(tmp_path: Path) -> None:
    link = tmp_path / "python"
    link.symlink_to(tmp_path / "deleted-env" / "bin" / "python3.12")
    path, failure = check(str(link), tmp_path)
    assert path is None
    assert failure.kind is FailureKind.NOT_FOUND
    assert "deleted-env" in failure.detail


@pending
@posix_only
def test_a_file_without_execute_permission_is_rejected(tmp_path: Path) -> None:
    program = fake_python(tmp_path)
    program.chmod(0o644)
    path, failure = check(str(program), tmp_path)
    assert path is None
    assert failure.kind is FailureKind.PERMISSION_DENIED


@pending
def test_an_empty_path_is_rejected(tmp_path: Path) -> None:
    path, failure = check("  ", tmp_path)
    assert path is None
    assert failure.kind is FailureKind.NOT_FOUND
