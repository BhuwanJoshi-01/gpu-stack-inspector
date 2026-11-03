"""Checks of the CUDA toolchain against what the driver supports."""

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


@RULES.rule("GSI-CUDA-001", "nvcc is installed")
def nvcc_installed(snapshot: Snapshot, context: RuleContext) -> Iterator[Finding]:
    """Whether the CUDA compiler nvcc is on PATH. Missing is not a problem by itself.

    Prebuilt PyTorch wheels bring their own CUDA runtime and do not need a local CUDA
    toolkit. nvcc only matters for compiling CUDA code yourself, such as building an
    extension from source.
    """
    toolkit = snapshot.toolkit
    if toolkit is None:
        return
    if toolkit.nvcc_path is not None:
        yield Finding(
            id="GSI-CUDA-001",
            status=Status.PASS,
            severity=Severity.INFO,
            confidence=Confidence.HIGH,
            summary=f"nvcc {toolkit.nvcc_version or '(unknown version)'} is installed",
            reason=f"Found at {toolkit.nvcc_path}.",
            evidence=(Evidence("nvcc", "version", toolkit.nvcc_version),),
        )
        return
    yield Finding(
        id="GSI-CUDA-001",
        status=Status.NOT_APPLICABLE,
        severity=Severity.INFO,
        confidence=Confidence.HIGH,
        summary="No CUDA compiler (nvcc) on PATH",
        reason=(
            "Prebuilt PyTorch wheels do not need it; it is only needed to compile CUDA code, "
            "such as building an extension from source."
        ),
        evidence=(Evidence("nvcc", "path", None),),
    )
