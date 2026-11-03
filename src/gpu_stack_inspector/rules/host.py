"""Checks of the machine: driver tools, the driver, the GPUs, variables that hide GPUs."""

from __future__ import annotations

from collections.abc import Iterator

from gpu_stack_inspector.models import (
    Confidence,
    Evidence,
    Finding,
    Severity,
    Snapshot,
    Status,
)
from gpu_stack_inspector.rules.catalog import RULES
from gpu_stack_inspector.rules.engine import RuleContext

SMI = "nvidia-smi"


@RULES.rule("GSI-SYS-001", "NVIDIA driver tools are installed")
def driver_tools(snapshot: Snapshot, context: RuleContext) -> Iterator[Finding]:
    """Whether nvidia-smi, which every NVIDIA driver installs, is on PATH.

    Without it the driver and the GPUs cannot be checked. Its absence usually means no
    NVIDIA driver is installed, but some container images run GPU workloads without it,
    so this is a warning, not a failure. When CUDA is not expected it does not apply.
    """
    nvidia = snapshot.nvidia
    if nvidia is None:
        return
    if nvidia.smi_path is not None:
        yield Finding(
            id="GSI-SYS-001",
            status=Status.PASS,
            severity=Severity.INFO,
            confidence=Confidence.HIGH,
            summary="nvidia-smi is installed",
            reason=f"Found at {nvidia.smi_path}.",
            evidence=(Evidence(SMI, "path", nvidia.smi_path),),
        )
        return
    yield Finding(
        id="GSI-SYS-001",
        status=Status.WARN if context.cuda_expected else Status.NOT_APPLICABLE,
        severity=Severity.MEDIUM if context.cuda_expected else Severity.INFO,
        confidence=Confidence.MEDIUM,
        summary="nvidia-smi was not found",
        reason=(
            "Without NVIDIA's driver tools on PATH the driver and the GPUs cannot be "
            "checked. A system with a working NVIDIA driver normally provides nvidia-smi."
        ),
        evidence=(Evidence(SMI, "path", None),),
        recommended_action=(
            "Install the NVIDIA driver for your distribution (it provides nvidia-smi), or "
            "add its directory to PATH. On WSL it is in /usr/lib/wsl/lib."
        ),
        verify_command="nvidia-smi",
        limitations=("Some container images run GPU workloads without nvidia-smi installed.",),
    )
