"""The classifier on probe answers captured from real environments."""

from __future__ import annotations

import pytest

from gpu_stack_inspector.models import Confidence
from gpu_stack_inspector.probes.metadata_probe import parse_probe_output
from gpu_stack_inspector.rules.torch_build import TorchBuildKind, classify_torch
from tests.fakes import read_fixture


@pytest.mark.parametrize(
    ("fixture", "kind", "cuda"),
    [
        ("conda-sapiens2.stdout.txt", TorchBuildKind.CUDA, "12.8"),
        ("conda-wilor.stdout.txt", TorchBuildKind.CUDA, "13.0"),
        ("conda-smplerx.stdout.txt", TorchBuildKind.CUDA, "11.3"),
        ("venv-cpu-torch.stdout.txt", TorchBuildKind.CPU, None),
        ("conda-base.stdout.txt", TorchBuildKind.MISSING, None),
        ("conda-dwpose.stdout.txt", TorchBuildKind.MISSING, None),
    ],
)
def test_real_environments_are_classified_with_high_confidence(
    fixture: str, kind: TorchBuildKind, cuda: str | None
) -> None:
    facts = parse_probe_output("A", "python", read_fixture("probe", fixture))
    build = classify_torch(facts.torch)
    assert (build.kind, build.cuda_version, build.confidence) == (kind, cuda, Confidence.HIGH)
    assert build.conflicts == ()
