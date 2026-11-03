from gpu_stack_inspector import __version__
from gpu_stack_inspector.models import EnvironmentVariable, FailureKind, HostFacts
from gpu_stack_inspector.orchestrator import inspect_host
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
