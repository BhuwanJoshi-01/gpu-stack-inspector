"""The interpreter checks on probe answers captured from real environments."""

from __future__ import annotations

import pytest

from gpu_stack_inspector.models import NvidiaFacts, Snapshot, Status
from gpu_stack_inspector.probes.metadata_probe import parse_probe_output
from gpu_stack_inspector.rules.catalog import builtin_rules
from gpu_stack_inspector.rules.engine import evaluate
from tests.fakes import read_fixture

DRIVER = NvidiaFacts("/usr/bin/nvidia-smi", "595.99.02", "13.2")


@pytest.mark.parametrize(
    ("fixture", "torch_installed", "torch_build"),
    [
        ("conda-sapiens2.stdout.txt", Status.PASS, Status.PASS),
        ("conda-wilor.stdout.txt", Status.PASS, Status.PASS),
        ("conda-smplerx.stdout.txt", Status.PASS, Status.PASS),
        ("venv-cpu-torch.stdout.txt", Status.PASS, Status.FAIL),
        ("conda-base.stdout.txt", Status.WARN, None),
        ("conda-dwpose.stdout.txt", Status.NOT_APPLICABLE, None),
    ],
)
def test_real_environments(
    fixture: str, torch_installed: Status, torch_build: Status | None
) -> None:
    interpreter = parse_probe_output("A", "python", read_fixture("probe", fixture))
    snapshot = Snapshot("0.1.0", "t", nvidia=DRIVER, interpreters=(interpreter,))
    statuses = {f.id: f.status for f in evaluate(snapshot, builtin_rules()) if f.subject == "A"}
    assert statuses["GSI-ENV-001"] is Status.PASS
    assert statuses["GSI-TORCH-001"] is torch_installed
    assert statuses.get("GSI-TORCH-002") is torch_build
