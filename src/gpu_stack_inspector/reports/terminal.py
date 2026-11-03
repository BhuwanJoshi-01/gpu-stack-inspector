"""Terminal output: a short human summary of an already redacted snapshot.

Collected text is always added as plain ``Text``, never as Rich markup, so a path or GPU
name containing ``[bold]`` is printed as it is instead of being interpreted.
"""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

from gpu_stack_inspector.models import (
    HostFacts,
    InterpreterFacts,
    NvidiaFacts,
    Snapshot,
    ToolkitFacts,
)

_INDENT = "         "


def render_terminal(snapshot: Snapshot, console: Console) -> None:
    console.print(Text(f"GPU Stack Inspector {snapshot.tool_version}", style="bold"))
    console.print(_row("Host", _host(snapshot.host)))
    nvidia_lines = _nvidia(snapshot.nvidia)
    console.print(_row("NVIDIA", nvidia_lines[0]))
    for line in nvidia_lines[1:]:
        console.print(Text(_INDENT + line))
    console.print(_row("Toolkit", _toolkit(snapshot.toolkit)))
    if snapshot.cuda_environment:
        values = ", ".join(
            f"{v.name}={v.value}" if v.value else f'{v.name}="" (hides every GPU)'
            for v in snapshot.cuda_environment
        )
        console.print(_row("CUDA env", values))
    for interpreter in snapshot.interpreters:
        console.print()
        console.print(Text.assemble((f"{interpreter.label:<3}", "bold"), interpreter.path))
        console.print(Text("   " + _interpreter(interpreter)))
    if snapshot.probe_failures:
        console.print()
        console.print(Text("Could not check:", style="bold"))
        for failure in snapshot.probe_failures:
            console.print(Text(f"  {failure.probe}: {failure.detail}"))


def _row(name: str, value: str) -> Text:
    return Text.assemble((f"{name:<9}", "bold"), value)


def _host(host: HostFacts | None) -> str:
    if host is None:
        return "not inspected"
    parts = [f"{host.system} {host.machine}"]
    if host.distribution:
        parts.append(host.distribution)
    if host.kernel:
        parts.append(f"kernel {host.kernel}")
    if host.wsl:
        parts.append("WSL")
    return " · ".join(parts)


def _nvidia(nvidia: NvidiaFacts | None) -> list[str]:
    if nvidia is None:
        return ["not inspected"]
    if nvidia.smi_path is None:
        return ["nvidia-smi not found (no NVIDIA driver tools on PATH)"]
    if nvidia.driver_version is None and not nvidia.gpus:
        return ["nvidia-smi did not answer (see below)"]
    summary = (
        f"driver {nvidia.driver_version or 'unknown'} · "
        f"supports CUDA up to {nvidia.max_cuda_version or 'unknown'}"
    )
    gpus = [
        f"GPU {gpu.index}: {gpu.name}"
        + f" ({gpu.memory_total_mib if gpu.memory_total_mib is not None else '?'} MiB,"
        + f" compute {gpu.compute_capability or 'unknown'})"
        for gpu in nvidia.gpus
    ]
    return [summary, *(gpus or ["no GPU visible to the driver"])]


def _toolkit(toolkit: ToolkitFacts | None) -> str:
    if toolkit is None:
        return "not inspected"
    nvcc = (
        f"nvcc {toolkit.nvcc_version or 'unknown version'} ({toolkit.nvcc_path})"
        if toolkit.nvcc_path
        else "no nvcc on PATH"
    )
    directories = ", ".join(
        f"{t.path} ({t.version or 'unknown version'})" for t in toolkit.toolkits
    )
    return f"{nvcc} · {directories or 'no CUDA toolkit directories found'}"


def _interpreter(interpreter: InterpreterFacts) -> str:
    if interpreter.version is None:
        return "could not be inspected (see below)"
    kind = [interpreter.implementation or "python"]
    if interpreter.conda:
        kind.append("conda")
    if interpreter.virtualenv:
        kind.append("venv")
    torch = next((d for d in interpreter.distributions if d.name == "torch"), None)
    wheels = sum(d.name.startswith("nvidia-") for d in interpreter.distributions)
    parts = [
        f"Python {interpreter.version} ({', '.join(kind)})",
        f"torch {torch.version}" if torch else "no torch",
        f"{wheels} nvidia-* wheels",
    ]
    return " · ".join(parts)
