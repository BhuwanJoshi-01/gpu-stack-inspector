import json

from gpu_stack_inspector.models import (
    Distribution,
    FailureKind,
    GpuDevice,
    InterpreterFacts,
    NvidiaFacts,
    ProbeFailure,
    Snapshot,
)
from gpu_stack_inspector.reports.json_report import render_json, snapshot_to_dict

SNAPSHOT = Snapshot(
    tool_version="0.1.0",
    created_at="2026-11-03T12:00:00Z",
    nvidia=NvidiaFacts(
        smi_path="/usr/bin/nvidia-smi",
        driver_version="595.99.02",
        max_cuda_version="13.2",
        gpus=(GpuDevice(0, "NVIDIA GeForce RTX 3050 Laptop GPU", 4096, "8.6"),),
    ),
    interpreters=(
        InterpreterFacts(
            "A",
            "~/envs/a/bin/python",
            version="3.12.13",
            prefix="~/envs/a",
            base_prefix="/usr",
            distributions=(Distribution("torch", "2.11.0+cu128", "~/envs/a/lib/site-packages"),),
        ),
    ),
    probe_failures=(ProbeFailure("nvcc", FailureKind.TIMEOUT, "nvcc did not finish"),),
)


def test_json_output_round_trips_to_plain_data() -> None:
    data = json.loads(render_json(SNAPSHOT))
    assert data == snapshot_to_dict(SNAPSHOT)
    assert data["schema_version"] == 1
    assert data["nvidia"]["gpus"][0] == {
        "index": 0,
        "name": "NVIDIA GeForce RTX 3050 Laptop GPU",
        "memory_total_mib": 4096,
        "compute_capability": "8.6",
    }


def test_enums_become_their_documented_strings() -> None:
    assert snapshot_to_dict(SNAPSHOT)["probe_failures"] == [
        {"probe": "nvcc", "kind": "timeout", "detail": "nvcc did not finish"}
    ]


def test_interpreters_include_whether_they_are_a_virtualenv() -> None:
    interpreter = snapshot_to_dict(SNAPSHOT)["interpreters"][0]
    assert interpreter["virtualenv"] is True
    assert interpreter["distributions"] == [
        {"name": "torch", "version": "2.11.0+cu128", "location": "~/envs/a/lib/site-packages"}
    ]


def test_output_ends_with_a_newline_and_keeps_unicode_readable() -> None:
    snapshot = Snapshot(
        tool_version="0.1.0", created_at="t", interpreters=(InterpreterFacts("A", "~/Ünï"),)
    )
    text = render_json(snapshot)
    assert text.endswith("}\n")
    assert "~/Ünï" in text
