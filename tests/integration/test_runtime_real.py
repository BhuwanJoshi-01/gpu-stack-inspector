"""The runtime probe on real interpreters with torch. Opt in with GSI_RUNTIME_PYTHONS.

This imports torch in each named interpreter and runs a tiny CUDA operation, like
`probe --runtime`. CI has no GPU and does not set the variable.
"""

from __future__ import annotations

import os

import pytest

from gpu_stack_inspector.probes.runtime_probe import run_runtime_probe
from gpu_stack_inspector.runner import SubprocessRunner

PYTHONS = [path for path in os.environ.get("GSI_RUNTIME_PYTHONS", "").split(os.pathsep) if path]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not PYTHONS, reason="set GSI_RUNTIME_PYTHONS to interpreters with torch"),
]


@pytest.mark.parametrize("python", PYTHONS)
def test_the_runtime_probe_gives_a_coherent_answer(python: str) -> None:
    facts = run_runtime_probe(SubprocessRunner(), python)
    assert facts.failure is None, facts.failure
    assert facts.torch_version
    assert facts.steps[0].step == "import torch"
    if facts.cuda_available:
        assert facts.devices
        assert facts.tensor_ok is True
    else:
        assert facts.tensor_ok is None
