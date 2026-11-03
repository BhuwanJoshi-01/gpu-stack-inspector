"""The installed command and ``python -m`` start the same CLI."""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path

import pytest

from gpu_stack_inspector import __version__
from gpu_stack_inspector.runner import Outcome, SubprocessRunner


def _console_script() -> Path:
    name = "gpu-stack-inspector.exe" if sys.platform == "win32" else "gpu-stack-inspector"
    return Path(sysconfig.get_path("scripts")) / name


@pytest.mark.parametrize(
    "argv",
    [
        [sys.executable, "-m", "gpu_stack_inspector", "--version"],
        [str(_console_script()), "--version"],
    ],
    ids=["python -m", "console script"],
)
def test_entry_point_prints_the_version(argv: list[str]) -> None:
    result = SubprocessRunner().run(argv, timeout=30)
    assert result.outcome is Outcome.OK, result.stderr
    assert result.stdout.strip() == f"gpu-stack-inspector {__version__}"
