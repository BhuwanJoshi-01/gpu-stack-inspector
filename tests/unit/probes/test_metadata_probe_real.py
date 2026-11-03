"""The probe against real interpreters: the test's own, a fresh venv, odd paths."""

from __future__ import annotations

import os
import sys
import venv
from pathlib import Path

import pytest

from gpu_stack_inspector.probes.metadata_probe import probe_interpreter
from gpu_stack_inspector.runner import Outcome, SubprocessRunner

THIS_PYTHON = ".".join(str(part) for part in sys.version_info[:3])


def test_a_module_planted_in_the_working_directory_is_not_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "json.py").write_text("raise SystemExit('planted json.py was imported')\n")
    monkeypatch.chdir(tmp_path)
    # Control: a plain `python -c` started here really would import the planted module.
    control = SubprocessRunner().run([sys.executable, "-c", "import json"], timeout=30)
    assert control.outcome is Outcome.NONZERO
    assert "planted json.py was imported" in control.stderr

    facts = probe_interpreter(SubprocessRunner(), "A", sys.executable)
    assert facts.failures == ()
    assert facts.version == THIS_PYTHON


def test_a_fresh_venv_is_seen_as_a_venv(tmp_path: Path) -> None:
    env_dir = tmp_path / "env"
    venv.create(env_dir, with_pip=False, symlinks=os.name != "nt")
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    facts = probe_interpreter(SubprocessRunner(), "venv", str(python))
    assert facts.failures == ()
    assert facts.virtualenv is True
    assert Path(facts.prefix or "").resolve() == env_dir.resolve()
    assert facts.version == THIS_PYTHON


@pytest.mark.skipif(
    os.name != "posix", reason="symlinking the interpreter needs privileges on Windows"
)
def test_a_path_with_spaces_and_unicode_works(tmp_path: Path) -> None:
    link = tmp_path / "envs" / "Ünïcode env ✓" / "bin" / "python"
    link.parent.mkdir(parents=True)
    link.symlink_to(sys.executable)
    facts = probe_interpreter(SubprocessRunner(), "A", str(link))
    assert facts.failures == ()
    assert facts.path == str(link)
