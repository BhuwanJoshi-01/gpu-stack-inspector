"""Checks across the inspected interpreters: what tells them apart."""

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
from gpu_stack_inspector.rules.comparison import torch_build_label
from gpu_stack_inspector.rules.engine import RuleContext

COMPARISON = "comparison"
"""Subject of findings about the interpreters as a group."""


@RULES.rule("GSI-DIFF-001", "The compared environments have the same PyTorch build")
def same_torch_build(snapshot: Snapshot, context: RuleContext) -> Iterator[Finding]:
    """Whether every compared environment has the same kind of PyTorch build.

    When one environment has a CUDA build and another a CPU-only build (or no torch, or a
    build for another CUDA version), that difference is the most likely reason the GPU
    works in one and not the other. Only interpreters that could be inspected count, and
    only when at least two of them are compared.
    """
    usable = [i for i in snapshot.interpreters if i.version is not None]
    if len(usable) < 2:
        return
    builds = [(i.label, torch_build_label(i) or "unknown") for i in usable]
    evidence = tuple(Evidence("comparison", label, build) for label, build in builds)
    listing = ", ".join(f"{label} {build}" for label, build in builds)
    if len({build for _, build in builds}) == 1:
        yield Finding(
            id="GSI-DIFF-001",
            status=Status.PASS,
            severity=Severity.INFO,
            confidence=Confidence.HIGH,
            summary=f"All compared environments have the same PyTorch build ({builds[0][1]})",
            reason="The difference, if any, is elsewhere; see the comparison table.",
            subject=COMPARISON,
            evidence=evidence,
        )
        return
    yield Finding(
        id="GSI-DIFF-001",
        status=Status.WARN,
        severity=Severity.MEDIUM,
        confidence=Confidence.HIGH,
        summary=f"The environments have different PyTorch builds: {listing}",
        reason=("This is the most likely reason CUDA works in one environment and not in another."),
        subject=COMPARISON,
        evidence=evidence,
        recommended_action=(
            "Install the same CUDA build of torch in the environment that should use the GPU; "
            "GSI-TORCH-002 says which build each one has."
        ),
        verify_command="gpu-stack-inspector compare --python … --python …",
    )
