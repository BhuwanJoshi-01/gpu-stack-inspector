import json
from dataclasses import replace
from pathlib import Path

from gpu_stack_inspector import __version__
from gpu_stack_inspector.models import EnvironmentVariable, FailureKind, HostFacts
from gpu_stack_inspector.orchestrator import (
    default_interpreters,
    diagnose,
    inspect,
    inspect_host,
    interpreter_label,
    with_comparison,
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


def test_diagnose_attaches_the_findings_of_the_builtin_rules() -> None:
    snapshot = inspect_host(
        FakeRunner(), which=lambda _: None, environ={}, toolkit_roots=(), host=lambda: HOST
    )
    diagnosed = diagnose(snapshot)
    assert [f.id for f in diagnosed.findings][:3] == ["GSI-SYS-001", "GSI-SYS-002", "GSI-SYS-003"]
    assert diagnosed.nvidia == snapshot.nvidia


def test_with_comparison_compares_the_inspected_interpreters() -> None:
    from gpu_stack_inspector.models import InterpreterFacts

    snapshot = inspect_host(
        FakeRunner(), which=lambda _: None, environ={}, toolkit_roots=(), host=lambda: HOST
    )
    snapshot = replace(
        snapshot,
        interpreters=(
            InterpreterFacts("A", "/a/python", version="3.12.0"),
            InterpreterFacts("B", "/b/python", version="3.13.0"),
        ),
    )
    compared = with_comparison(snapshot)
    python_row = next(row for row in compared.comparison if row.field == "Python")
    assert python_row.values == ("3.12.0", "3.13.0")


def _script_metadata(runner: FakeRunner, python: Path, torch: bool) -> None:
    document = {
        "protocol": 1,
        "errors": [],
        "python": {"version": "3.12.13"},
        "distributions": [],
        "torch": {"version": "2.11.0+cu128"} if torch else None,
    }
    runner.script(
        str(python),
        "-c",
        program_source(),
        stdout=f"GSI-PROBE-BEGIN{json.dumps(document)}GSI-PROBE-END",
    )


def _interpreter(tmp_path: Path, name: str) -> Path:
    python = tmp_path / name / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    return python


def test_the_runtime_probe_runs_only_with_consent_and_only_where_torch_is(tmp_path: Path) -> None:
    from gpu_stack_inspector.probes import runtime_probe

    with_torch, without_torch = _interpreter(tmp_path, "a"), _interpreter(tmp_path, "b")
    runner = FakeRunner()
    _script_metadata(runner, with_torch, torch=True)
    _script_metadata(runner, without_torch, torch=False)
    runner.script(
        str(with_torch),
        "-c",
        runtime_probe.program_source(),
        stdout=read_fixture("runtime", "sapiens2.stdout.txt"),
    )
    kwargs = {"which": lambda _: None, "environ": {}, "toolkit_roots": (), "host": lambda: HOST}

    plain = inspect(runner, [str(with_torch), str(without_torch)], **kwargs)  # type: ignore[arg-type]
    assert [i.runtime for i in plain.interpreters] == [None, None]
    assert not any(runtime_probe.program_source() in call.argv for call in runner.calls)

    probed = inspect(runner, [str(with_torch), str(without_torch)], runtime=True, **kwargs)  # type: ignore[arg-type]
    first, second = probed.interpreters
    assert first.runtime is not None
    assert first.runtime.tensor_ok is True
    assert second.runtime is None
