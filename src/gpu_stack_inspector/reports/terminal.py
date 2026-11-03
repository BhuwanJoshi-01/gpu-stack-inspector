"""Terminal output: a short human summary of an already redacted snapshot.

Collected text is always added as plain ``Text``, never as Rich markup, so a path or GPU
name containing ``[bold]`` is printed as it is instead of being interpreted.
"""

from __future__ import annotations

from collections import Counter

from rich.console import Console
from rich.table import Table
from rich.text import Text

from gpu_stack_inspector.models import (
    Finding,
    HostFacts,
    InterpreterFacts,
    NvidiaFacts,
    Snapshot,
    Status,
    ToolkitFacts,
)
from gpu_stack_inspector.rules.torch_build import TorchBuild, TorchBuildKind, classify_torch

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
    if snapshot.findings:
        _render_findings(snapshot, console)
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
    wheels = sum(d.name.startswith("nvidia-") for d in interpreter.distributions)
    parts = [
        f"Python {interpreter.version} ({', '.join(kind)})",
        _torch(interpreter),
        f"{wheels} nvidia-* wheels",
    ]
    return " · ".join(parts)


def _torch(interpreter: InterpreterFacts) -> str:
    if interpreter.torch is None:
        listed = next((d for d in interpreter.distributions if d.name == "torch"), None)
        return f"torch {listed.version} (build not inspected)" if listed else "no torch"
    return f"torch {interpreter.torch.version} ({_build(classify_torch(interpreter.torch))})"


def _build(build: TorchBuild) -> str:
    """Say what the metadata shows, and only that: no CUDA operation has run yet."""
    if build.kind is TorchBuildKind.CUDA:
        target = f"CUDA {build.cuda_version}" if build.cuda_version else "CUDA"
        text = f"{target} build"
    elif build.kind is TorchBuildKind.CPU:
        text = "CPU-only build"
    elif build.kind is TorchBuildKind.ROCM:
        text = "ROCm build, not checked by this tool"
    else:
        text = "build unknown"
    if build.conflicts:
        text += f"; {build.conflicts[0]}"
    elif build.kind is not TorchBuildKind.UNKNOWN and build.confidence.value != "high":
        text += f", {build.confidence.value} confidence"
    return text


_LABELS = {
    Status.FAIL: ("FAIL", "bold red"),
    Status.WARN: ("WARN", "yellow"),
    Status.UNKNOWN: ("UNKNOWN", "magenta"),
    Status.PASS: ("PASS", "green"),
    Status.NOT_APPLICABLE: ("N/A", "dim"),
    Status.NOT_TESTED: ("NOT TESTED", "dim"),
}
_PROBLEMS = frozenset({Status.FAIL, Status.WARN, Status.UNKNOWN})


def _render_findings(snapshot: Snapshot, console: Console) -> None:
    """Findings per subject (host, then each interpreter) in evaluation order.

    Problems show their reason, the next step and how to check it; consequences of a
    root cause are dimmed and say which finding they follow from.
    """
    paths = {interpreter.label: interpreter.path for interpreter in snapshot.interpreters}
    subjects = ["host", *paths, "comparison"]
    console.print()
    console.print(Text("Findings", style="bold"))
    for subject in subjects:
        group = [f for f in snapshot.findings if f.subject == subject]
        if not group:
            continue
        heading = (
            "Host"
            if subject == "host"
            else "Comparison"
            if subject == "comparison"
            else f"{subject}  {paths[subject]}"
        )
        console.print(Text(heading, style="bold"))
        table = Table.grid(padding=(0, 1))
        table.add_column(width=10, no_wrap=True)
        table.add_column(width=13, no_wrap=True)  # GSI-TORCH-002 is the longest code
        # Fold, never crop: people copy the commands and paths shown here.
        table.add_column(overflow="fold")
        for finding in group:
            _add_finding(table, finding)
        console.print(table)
    console.print(Text(_tally(snapshot.findings)))


def _add_finding(table: Table, finding: Finding) -> None:
    label, style = _LABELS[finding.status]
    summary = finding.summary
    if finding.caused_by:
        summary += f" (follows from {finding.caused_by})"
        style = "dim"
    table.add_row(Text(label, style=style), Text(finding.id, style="dim"), Text(summary))
    if finding.status not in _PROBLEMS:
        return
    details = [finding.reason]
    if finding.recommended_action:
        details.append(f"Next:  {finding.recommended_action}")
    if finding.verify_command:
        details.append(f"Check: {finding.verify_command}")
    details.extend(f"Note:  {limitation}" for limitation in finding.limitations)
    for detail in details:
        table.add_row(Text(""), Text(""), Text(detail))


def _tally(findings: tuple[Finding, ...]) -> str:
    counts = Counter(f.status for f in findings)
    parts = [
        f"{counts[Status.FAIL]} failed",
        f"{counts[Status.WARN]} warning{'' if counts[Status.WARN] == 1 else 's'}",
        f"{counts[Status.UNKNOWN]} unknown",
        f"{counts[Status.PASS]} passed",
        f"{counts[Status.NOT_APPLICABLE] + counts[Status.NOT_TESTED]} not applicable or not tested",
    ]
    return "Summary: " + " · ".join(parts)
