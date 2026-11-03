"""Compatibility checks on this laptop's captured driver and environments."""

from __future__ import annotations

from gpu_stack_inspector.models import NvidiaFacts, Snapshot, Status, ToolkitFacts
from gpu_stack_inspector.probes.metadata_probe import parse_probe_output
from gpu_stack_inspector.rules.catalog import builtin_rules
from gpu_stack_inspector.rules.engine import evaluate
from tests.fakes import read_fixture

DRIVER = NvidiaFacts("/usr/bin/nvidia-smi", "595.99.02", "13.2")
NVCC = ToolkitFacts("/usr/bin/nvcc", "12.4.131")


def test_driver_13_2_runs_every_real_cuda_build_and_the_local_nvcc() -> None:
    interpreters = tuple(
        parse_probe_output(label, "python", read_fixture("probe", fixture))
        for label, fixture in (
            ("A", "conda-sapiens2.stdout.txt"),
            ("B", "conda-wilor.stdout.txt"),
            ("C", "conda-smplerx.stdout.txt"),
            ("D", "venv-cpu-torch.stdout.txt"),
        )
    )
    snapshot = Snapshot("0.1.0", "t", nvidia=DRIVER, toolkit=NVCC, interpreters=interpreters)
    findings = evaluate(snapshot, builtin_rules())
    torch_vs_driver = {f.subject: (f.status, f.summary) for f in findings if f.id == "GSI-CUDA-003"}
    assert torch_vs_driver == {
        "A": (Status.PASS, "The driver supports CUDA 12.8"),
        "B": (Status.PASS, "The driver supports CUDA 13.0"),
        "C": (Status.PASS, "The driver supports CUDA 11.3"),
    }
    [nvcc] = [f for f in findings if f.id == "GSI-CUDA-002"]
    assert (nvcc.status, nvcc.summary) == (Status.PASS, "The driver supports nvcc's CUDA 12.4")
    assert not [f for f in findings if "internal error" in f.summary]
