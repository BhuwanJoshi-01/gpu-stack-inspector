"""Checks of the CUDA toolchain against what the driver supports."""

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum

from gpu_stack_inspector.compatibility.loader import load
from gpu_stack_inspector.compatibility.versions import major_minor
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
from gpu_stack_inspector.rules.torch_build import TorchBuildKind, classify_torch

_MINOR_URL = "https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html"


class _Support(Enum):
    SUPPORTED = "supported"
    NEWER_MINOR = "newer minor"
    NEWER_MAJOR = "newer major"


def _support(target: tuple[int, int], driver_max: tuple[int, int]) -> _Support:
    if target <= driver_max:
        return _Support.SUPPORTED
    if target[0] == driver_max[0] and target[0] >= load().minor_version_from_major:
        return _Support.NEWER_MINOR  # minor version compatibility applies
    return _Support.NEWER_MAJOR


def _driver_cause(snapshot: Snapshot) -> str | None:
    """Which host finding explains a missing driver CUDA version, if any."""
    nvidia = snapshot.nvidia
    if nvidia is not None and nvidia.smi_path is None:
        return "GSI-SYS-001"
    if nvidia is not None and nvidia.driver_version is None:
        return "GSI-SYS-002"
    return None


def _not_compared(snapshot: Snapshot, rule_id: str, what: str, subject: str = "host") -> Finding:
    """No driver CUDA version: not tested because of a host finding, or plainly unknown."""
    cause = _driver_cause(snapshot)
    if cause is not None:
        return Finding(
            id=rule_id,
            status=Status.NOT_TESTED,
            severity=Severity.INFO,
            confidence=Confidence.HIGH,
            summary=f"{what} was not compared with the driver",
            reason="The driver's CUDA version is not known.",
            subject=subject,
            caused_by=cause,
        )
    return Finding(
        id=rule_id,
        status=Status.UNKNOWN,
        severity=Severity.LOW,
        confidence=Confidence.LOW,
        summary=f"{what} could not be compared with the driver",
        reason="The driver answered but did not report the newest CUDA version it supports.",
        subject=subject,
        recommended_action="Check the 'CUDA Version' in the header of `nvidia-smi` by hand.",
        verify_command="nvidia-smi",
    )


def _minimum_driver_sentence(major: int) -> str:
    row = load().minimum_driver.get(major)
    return f" CUDA {major}.x needs driver {row[0]} or newer." if row else ""


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


@RULES.rule("GSI-CUDA-002", "The driver supports nvcc's CUDA version", sources=(_MINOR_URL,))
def nvcc_supported(snapshot: Snapshot, context: RuleContext) -> Iterator[Finding]:
    """Whether programs compiled with the local nvcc can run on this driver.

    nvcc from CUDA 13.x produces programs that need a driver supporting CUDA 13. Within one
    major release family, NVIDIA's minor version compatibility usually lets newer-minor
    programs run, with a limited feature set; code compiled to PTX does not.
    """
    toolkit, nvidia = snapshot.toolkit, snapshot.nvidia
    nvcc = major_minor(toolkit.nvcc_version) if toolkit else None
    if nvcc is None:
        return
    driver_max = major_minor(nvidia.max_cuda_version) if nvidia else None
    if driver_max is None:
        if nvidia is not None:
            yield _not_compared(snapshot, "GSI-CUDA-002", "nvcc")
        return
    version = toolkit.nvcc_version if toolkit else None
    driver_text = nvidia.max_cuda_version if nvidia else None
    evidence = (
        Evidence("nvcc", "version", version),
        Evidence("nvidia-smi", "max_cuda_version", driver_text),
    )
    support = _support(nvcc, driver_max)
    if support is _Support.SUPPORTED:
        yield Finding(
            id="GSI-CUDA-002",
            status=Status.PASS,
            severity=Severity.INFO,
            confidence=Confidence.HIGH,
            summary=f"The driver supports nvcc's CUDA {nvcc[0]}.{nvcc[1]}",
            reason=f"The driver supports CUDA up to {driver_text}.",
            evidence=evidence,
        )
        return
    yield Finding(
        id="GSI-CUDA-002",
        status=Status.WARN,
        severity=Severity.MEDIUM if support is _Support.NEWER_MAJOR else Severity.LOW,
        confidence=Confidence.HIGH,
        summary=f"Programs built with nvcc {version} may need a newer driver",
        reason=(
            f"nvcc builds for CUDA {nvcc[0]}.{nvcc[1]}, but the driver supports CUDA up to "
            f"{driver_text}."
            + (
                _minimum_driver_sentence(nvcc[0])
                if support is _Support.NEWER_MAJOR
                else " Minor version compatibility usually lets them run, but code compiled "
                "to PTX will not."
            )
        ),
        evidence=evidence,
        recommended_action="Update the NVIDIA driver, or use a CUDA toolkit the driver supports.",
        verify_command="nvidia-smi",
    )


@RULES.rule(
    "GSI-CUDA-003",
    "The driver supports the CUDA version PyTorch was built for",
    sources=(_MINOR_URL,),
)
def torch_cuda_supported(snapshot: Snapshot, context: RuleContext) -> Iterator[Finding]:
    """Whether the driver can run the CUDA version each CUDA build of PyTorch targets.

    A build for a newer CUDA major release than the driver supports cannot initialise
    CUDA at all (unless NVIDIA's forward-compatibility package is installed, which is
    rare outside data centres). A newer minor version of the same release family usually
    works through minor version compatibility; operations that need the newer driver fail
    with cudaErrorCallRequiresNewerDriver.
    """
    nvidia = snapshot.nvidia
    driver_max = major_minor(nvidia.max_cuda_version) if nvidia else None
    driver_text = nvidia.max_cuda_version if nvidia else None
    for interpreter in snapshot.interpreters:
        if interpreter.torch is None:
            continue
        build = classify_torch(interpreter.torch)
        target = major_minor(build.cuda_version)
        if build.kind is not TorchBuildKind.CUDA or target is None:
            continue
        if driver_max is None:
            if nvidia is not None:
                what = f"CUDA {build.cuda_version}"
                yield _not_compared(snapshot, "GSI-CUDA-003", what, interpreter.label)
            continue
        evidence = (
            Evidence("torch", "cuda_version", build.cuda_version),
            Evidence("nvidia-smi", "max_cuda_version", driver_text),
        )
        support = _support(target, driver_max)
        if support is _Support.SUPPORTED:
            yield Finding(
                id="GSI-CUDA-003",
                status=Status.PASS,
                severity=Severity.INFO,
                confidence=Confidence.HIGH,
                summary=f"The driver supports CUDA {build.cuda_version}",
                reason=(
                    f"torch targets CUDA {build.cuda_version}; "
                    f"the driver supports up to {driver_text}."
                ),
                subject=interpreter.label,
                evidence=evidence,
            )
            continue
        newer_major = support is _Support.NEWER_MAJOR
        yield Finding(
            id="GSI-CUDA-003",
            status=Status.FAIL if newer_major else Status.WARN,
            severity=Severity.HIGH if newer_major else Severity.MEDIUM,
            confidence=Confidence.HIGH,
            summary=(
                f"torch is built for CUDA {build.cuda_version}, newer than the driver supports"
            ),
            reason=(
                f"The driver supports CUDA up to {driver_text}."
                + (
                    _minimum_driver_sentence(target[0]) + " CUDA will not initialise."
                    if newer_major
                    else " Minor version compatibility usually lets it run, but operations "
                    "that need a newer driver fail with cudaErrorCallRequiresNewerDriver."
                )
            ),
            subject=interpreter.label,
            evidence=evidence,
            recommended_action=(
                f"Update the NVIDIA driver, or install a torch build for CUDA {driver_text} or "
                "older from https://pytorch.org/get-started/locally/."
            ),
            verify_command="nvidia-smi",
            limitations=(
                (
                    "NVIDIA's forward-compatibility package (cuda-compat) can bridge major "
                    "versions on supported data-centre GPUs; it is not detected.",
                )
                if newer_major
                else ()
            ),
        )
