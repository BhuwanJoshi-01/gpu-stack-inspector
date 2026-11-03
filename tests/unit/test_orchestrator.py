import json
from pathlib import Path

from gpu_stack_inspector import __version__
from gpu_stack_inspector.models import EnvironmentVariable, FailureKind, HostFacts
from gpu_stack_inspector.orchestrator import (
    default_interpreters,
    inspect,
    inspect_host,
    interpreter_label,
)
from gpu_stack_inspector.probes.metadata_probe import program_source
from gpu_stack_inspector.runner import CommandResult, Outcome
from tests.fakes import FakeRunner, read_fixture

SMI, NVCC = "/usr/bin/nvidia-smi", "/usr/bin/nvcc"
HOST = HostFacts(
    system="Linux", machine="x86_64", kernel="7.0.0", distribution="Ubuntu 26.04.1 LTS"
)


def test_host_snapshot_combines_every_collector() -> None:
    runner = FakeRunner()
    runner.script(
        SMI,
        "--query-gpu=index,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
        stdout=read_fixture("nvidia-smi", "rtx3050-laptop-595.query.csv"),
    )
    runner.script(
        SMI, "--version", stdout=read_fixture("nvidia-smi", "rtx3050-laptop-595.version.txt")
    )
    runner.script_result(CommandResult((NVCC, "--version"), Outcome.TIMEOUT, duration=8))
    snapshot = inspect_host(
        runner,
        which={"nvidia-smi": SMI, "nvcc": NVCC}.get,
        environ={"CUDA_VISIBLE_DEVICES": "0", "HF_TOKEN": "secret"},
        toolkit_roots=(),
        host=lambda: HOST,
    )
    assert snapshot.tool_version == __version__
    assert snapshot.host == HOST
    assert snapshot.nvidia is not None
    assert snapshot.nvidia.max_cuda_version == "13.2"
    assert snapshot.toolkit is not None
    assert snapshot.cuda_environment == (EnvironmentVariable("CUDA_VISIBLE_DEVICES", "0"),)
    assert [(f.probe, f.kind) for f in snapshot.probe_failures] == [("nvcc", FailureKind.TIMEOUT)]
    assert snapshot.findings == ()


def test_a_machine_without_nvidia_tools_runs_no_commands() -> None:
    snapshot = inspect_host(
        FakeRunner(), which=lambda _: None, environ={}, toolkit_roots=(), host=lambda: HOST
    )
    assert snapshot.nvidia is not None
    assert snapshot.nvidia.smi_path is None
    assert snapshot.probe_failures == ()


def test_interpreters_are_labelled_in_order_and_their_failures_tagged(tmp_path: Path) -> None:
    good = tmp_path / "a" / "bin" / "python"
    good.parent.mkdir(parents=True)
    good.write_text("#!/bin/sh\n")
    good.chmod(0o755)
    runner = FakeRunner()
    document = {"protocol": 1, "errors": [], "python": {"version": "3.12.13"}, "distributions": []}
    runner.script(
        str(good),
        "-c",
        program_source(),
        stdout=f"GSI-PROBE-BEGIN{json.dumps(document)}GSI-PROBE-END",
    )
    snapshot = inspect(
        runner,
        [str(good), str(tmp_path / "missing" / "python")],
        which=lambda _: None,
        environ={},
        toolkit_roots=(),
        host=lambda: HOST,
    )
    assert [(i.label, i.version) for i in snapshot.interpreters] == [("A", "3.12.13"), ("B", None)]
    assert [(f.probe, f.kind) for f in snapshot.probe_failures] == [
        ("python B", FailureKind.NOT_FOUND)
    ]


def test_without_interpreters_only_the_host_is_inspected() -> None:
    snapshot = inspect(
        FakeRunner(), [], which=lambda _: None, environ={}, toolkit_roots=(), host=lambda: HOST
    )
    assert snapshot.interpreters == ()


def test_labels_run_from_a_to_z_and_then_number() -> None:
    assert [interpreter_label(i) for i in (0, 1, 25, 26)] == ["A", "B", "Z", "#27"]


def test_the_default_interpreter_is_the_python_on_path() -> None:
    found = {"python": "/envs/a/bin/python", "python3": "/usr/bin/python3"}
    assert default_interpreters(found.get) == ["/envs/a/bin/python"]
    assert default_interpreters({"python3": "/usr/bin/python3"}.get) == ["/usr/bin/python3"]
    assert default_interpreters(lambda _: None) == []
