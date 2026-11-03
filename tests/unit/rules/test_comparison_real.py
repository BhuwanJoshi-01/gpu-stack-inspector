"""Comparing real environments captured on one laptop."""

from __future__ import annotations

from gpu_stack_inspector.probes.metadata_probe import parse_probe_output
from gpu_stack_inspector.rules.comparison import compare_interpreters
from tests.fakes import read_fixture


def test_the_cu128_conda_env_against_the_cpu_only_venv() -> None:
    working = parse_probe_output(
        "A", "sapiens2", read_fixture("probe", "conda-sapiens2.stdout.txt")
    )
    broken = parse_probe_output("B", "cpu-venv", read_fixture("probe", "venv-cpu-torch.stdout.txt"))
    rows = {row.field: row for row in compare_interpreters([working, broken])}
    assert rows["torch build"].values == ("CUDA 12.8", "CPU only")
    assert rows["torch"].values == ("2.11.0+cu128", "2.15.0+cpu")
    assert rows["conda env"].values == ("yes", "no")
    assert rows["venv"].values == ("no", "yes")
    missing_wheels = [
        r.field for r in rows.values() if r.field.startswith("nvidia-") and r.values[1] is None
    ]
    assert len(missing_wheels) == 15  # every CUDA wheel of the working env is absent from the venv
