"""The metadata probe in other interpreters than the one running the tests.

Set GSI_TEST_PYTHONS to interpreter paths separated by os.pathsep (":" on POSIX). CI
installs CPython 3.8 to 3.10 and PyPy with uv and runs this; locally, point it at real
environments, e.g. conda envs.
"""

from __future__ import annotations

import os
import re

import pytest

from gpu_stack_inspector.probes.metadata_probe import probe_interpreter
from gpu_stack_inspector.runner import SubprocessRunner

PYTHONS = [path for path in os.environ.get("GSI_TEST_PYTHONS", "").split(os.pathsep) if path]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not PYTHONS, reason="set GSI_TEST_PYTHONS to interpreter paths"),
]


@pytest.mark.parametrize("python", PYTHONS)
def test_the_probe_answers_cleanly(python: str) -> None:
    facts = probe_interpreter(SubprocessRunner(), "A", python)
    assert facts.failures == (), facts.failures
    assert facts.version is not None
    assert facts.implementation in {"cpython", "pypy"}
    assert facts.executable


@pytest.mark.parametrize("python", PYTHONS)
def test_the_version_reported_is_the_interpreters_own(python: str) -> None:
    version = SubprocessRunner().run([python, "--version"], timeout=30)
    facts = probe_interpreter(SubprocessRunner(), "A", python)
    match = re.search(r"Python (\d+\.\d+\.\d+)", version.stdout + version.stderr)
    assert match is not None
    assert facts.version == match.group(1)
