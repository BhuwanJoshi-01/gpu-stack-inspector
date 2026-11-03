from rich.console import Console

from gpu_stack_inspector.models import (
    CudaToolkit,
    Distribution,
    EnvironmentVariable,
    FailureKind,
    GpuDevice,
    HostFacts,
    InterpreterFacts,
    NvidiaFacts,
    ProbeFailure,
    Snapshot,
    ToolkitFacts,
    TorchFacts,
)
from gpu_stack_inspector.reports.terminal import render_terminal


def render(snapshot: Snapshot) -> str:
    console = Console(record=True, width=120, color_system=None)
    render_terminal(snapshot, console)
    return console.export_text()


FULL = Snapshot(
    tool_version="0.1.0",
    created_at="t",
    host=HostFacts("Linux", "x86_64", "7.0.0-42-generic", "Ubuntu 26.04.1 LTS"),
    nvidia=NvidiaFacts(
        "/usr/bin/nvidia-smi",
        "595.99.02",
        "13.2",
        (GpuDevice(0, "NVIDIA GeForce RTX 3050 Laptop GPU", 4096, "8.6"),),
    ),
    toolkit=ToolkitFacts(
        "/usr/bin/nvcc", "12.4.131", (CudaToolkit("/usr/local/cuda-12.8", "12.8.0"),)
    ),
    cuda_environment=(EnvironmentVariable("CUDA_VISIBLE_DEVICES", ""),),
    interpreters=(
        InterpreterFacts(
            "A",
            "~/miniconda3/envs/sapiens2/bin/python",
            version="3.13.13",
            implementation="cpython",
            conda=True,
            distributions=(
                Distribution("torch", "2.11.0+cu128"),
                Distribution("nvidia-cublas-cu12", "12.8.4.1"),
            ),
            torch=TorchFacts(
                version="2.11.0+cu128",
                cuda_library=True,
                hip_library=False,
                version_py_found=True,
                built_cuda="12.8",
            ),
        ),
        InterpreterFacts("B", "~/gone/bin/python"),
    ),
    probe_failures=(
        ProbeFailure("python B", FailureKind.NOT_FOUND, "~/gone/bin/python does not exist"),
    ),
)


def test_a_full_snapshot_reads_as_a_short_summary() -> None:
    text = render(FULL)
    assert "GPU Stack Inspector 0.1.0" in text
    assert "Host     Linux x86_64 · Ubuntu 26.04.1 LTS · kernel 7.0.0-42-generic" in text
    assert "NVIDIA   driver 595.99.02 · supports CUDA up to 13.2" in text
    assert "GPU 0: NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB, compute 8.6)" in text
    assert "Toolkit  nvcc 12.4.131 (/usr/bin/nvcc) · /usr/local/cuda-12.8 (12.8.0)" in text
    assert 'CUDA_VISIBLE_DEVICES="" (hides every GPU)' in text
    assert (
        "Python 3.13.13 (cpython, conda) · torch 2.11.0+cu128 (CUDA 12.8 build) · 1 nvidia-* wheels"
        in text
    )
    assert "could not be inspected (see below)" in text
    assert "python B: ~/gone/bin/python does not exist" in text


def test_missing_nvidia_tools_are_described_not_failed() -> None:
    text = render(Snapshot("0.1.0", "t", nvidia=NvidiaFacts(None), toolkit=ToolkitFacts()))
    assert "nvidia-smi not found (no NVIDIA driver tools on PATH)" in text
    assert "no nvcc on PATH · no CUDA toolkit directories found" in text


def test_collected_text_is_never_interpreted_as_markup() -> None:
    snapshot = Snapshot(
        "0.1.0", "t", interpreters=(InterpreterFacts("A", "/envs/[bold red]x[/]/bin/python"),)
    )
    assert "/envs/[bold red]x[/]/bin/python" in render(snapshot)


def _interpreter_line(torch: TorchFacts | None) -> str:
    snapshot = Snapshot(
        "0.1.0", "t", interpreters=(InterpreterFacts("A", "p", version="3.12.0", torch=torch),)
    )
    return " ".join(render(snapshot).split())


def test_a_cpu_build_is_named_as_such() -> None:
    torch = TorchFacts(
        "2.15.0+cpu", cuda_library=False, hip_library=False, version_py_found=True, built_cuda=None
    )
    assert "torch 2.15.0+cpu (CPU-only build)" in _interpreter_line(torch)


def test_a_damaged_install_shows_the_conflict() -> None:
    torch = TorchFacts(
        "2.11.0+cu128",
        cuda_library=False,
        hip_library=False,
        version_py_found=True,
        built_cuda=None,
    )
    line = _interpreter_line(torch)
    assert "torch 2.11.0+cu128 (build unknown; the version tag +cu128 says cuda" in line


def test_weak_evidence_says_its_confidence() -> None:
    torch = TorchFacts("2.14.1", requires_nvidia=("nvidia-cudnn-cu13",))
    assert "torch 2.14.1 (CUDA build, low confidence)" in _interpreter_line(torch)


def test_rocm_is_named_but_not_judged() -> None:
    torch = TorchFacts("2.9.1+rocm6.4", hip_library=True, cuda_library=False)
    assert "ROCm build, not checked by this tool" in _interpreter_line(torch)


def test_without_torch_evidence_it_says_no_torch() -> None:
    assert "no torch" in _interpreter_line(None)
