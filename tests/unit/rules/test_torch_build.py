"""Classifying torch builds from static evidence (cases mirror real installs)."""

from __future__ import annotations

import pytest

from gpu_stack_inspector.models import Confidence, TorchFacts
from gpu_stack_inspector.rules.torch_build import TorchBuildKind, classify_torch

pending = pytest.mark.xfail(strict=True, reason="GPU-004: torch build classifier pending")

PYTORCH_INDEX_CU128 = TorchFacts(
    version="2.11.0+cu128",
    requires_nvidia=("nvidia-cudnn-cu12", "nvidia-nccl-cu12"),
    cuda_library=True,
    hip_library=False,
    version_py_found=True,
    built_cuda="12.8",
)
PYPI_UNTAGGED_CUDA = TorchFacts(
    version="2.14.1",
    requires_nvidia=("nvidia-cudnn-cu13",),
    cuda_library=True,
    hip_library=False,
    version_py_found=True,
    built_cuda="13.0",
)
CONDA_CUDA = TorchFacts(
    version="1.12.0",
    cuda_library=True,
    hip_library=False,
    version_py_found=True,
    built_cuda="11.3",
    conda_records=(
        "cudatoolkit-11.3.1-h2bc3f7f_2",
        "pytorch-1.12.0-py3.8_cuda11.3_cudnn8.3.2_0",
        "pytorch-mutex-1.0-cuda",
    ),
)
CPU_WHEEL = TorchFacts(
    version="2.11.0+cpu",
    cuda_library=False,
    hip_library=False,
    version_py_found=True,
    built_cuda=None,
)
ROCM_WHEEL = TorchFacts(
    version="2.9.1+rocm6.4",
    cuda_library=False,
    hip_library=True,
    version_py_found=True,
    built_hip="6.4.43482",
)


@pending
def test_no_torch_is_missing() -> None:
    build = classify_torch(None)
    assert build.kind is TorchBuildKind.MISSING
    assert build.confidence is Confidence.HIGH


@pending
@pytest.mark.parametrize(
    ("facts", "kind", "cuda"),
    [
        (PYTORCH_INDEX_CU128, TorchBuildKind.CUDA, "12.8"),
        (PYPI_UNTAGGED_CUDA, TorchBuildKind.CUDA, "13.0"),
        (CONDA_CUDA, TorchBuildKind.CUDA, "11.3"),
        (CPU_WHEEL, TorchBuildKind.CPU, None),
        (ROCM_WHEEL, TorchBuildKind.ROCM, None),
    ],
    ids=["pytorch-index-cu128", "pypi-untagged-cuda", "conda-cuda", "cpu-wheel", "rocm-wheel"],
)
def test_real_kinds_of_install_are_classified_with_high_confidence(
    facts: TorchFacts, kind: TorchBuildKind, cuda: str | None
) -> None:
    build = classify_torch(facts)
    assert (build.kind, build.cuda_version, build.confidence) == (kind, cuda, Confidence.HIGH)
    assert build.conflicts == ()
    assert build.evidence


@pending
@pytest.mark.parametrize(
    ("version", "cuda"),
    [
        ("2.5.1+cu121", "12.1"),
        ("2.0.1+cu118", "11.8"),
        ("1.7.1+cu92", "9.2"),
        ("2.11.0+cu128", "12.8"),
    ],
)
def test_the_cuda_version_is_read_from_the_tag_when_nothing_else_says(
    version: str, cuda: str
) -> None:
    build = classify_torch(TorchFacts(version=version))
    assert (build.kind, build.cuda_version, build.confidence) == (
        TorchBuildKind.CUDA,
        cuda,
        Confidence.MEDIUM,
    )


@pending
def test_a_conda_cpu_build_is_recognised_from_its_records() -> None:
    facts = TorchFacts(
        version="2.3.0",
        conda_records=("pytorch-2.3.0-py3.11_cpu_0", "pytorch-mutex-1.0-cpu"),
    )
    assert classify_torch(facts).kind is TorchBuildKind.CPU


@pending
def test_a_conda_forge_cuda_build_string_gives_the_version() -> None:
    facts = TorchFacts(version="2.4.1", conda_records=("pytorch-2.4.1-cuda118_py310h6b1e8c4_300",))
    build = classify_torch(facts)
    assert (build.kind, build.cuda_version) == (TorchBuildKind.CUDA, "11.8")


@pending
def test_nvidia_requirements_alone_are_weak_evidence() -> None:
    build = classify_torch(TorchFacts(version="2.14.1", requires_nvidia=("nvidia-cudnn-cu13",)))
    assert (build.kind, build.confidence) == (TorchBuildKind.CUDA, Confidence.LOW)


@pending
def test_no_evidence_at_all_is_unknown() -> None:
    build = classify_torch(TorchFacts(version="2.14.1"))
    assert build.kind is TorchBuildKind.UNKNOWN
    assert build.confidence is Confidence.LOW


@pending
def test_strong_signals_that_disagree_give_unknown_with_the_conflict() -> None:
    damaged = TorchFacts(
        version="2.11.0+cu128",
        cuda_library=False,
        hip_library=False,
        version_py_found=True,
        built_cuda=None,
    )
    build = classify_torch(damaged)
    assert build.kind is TorchBuildKind.UNKNOWN
    assert build.confidence is Confidence.LOW
    assert build.conflicts
    assert any("+cu128" in conflict for conflict in build.conflicts)


@pending
def test_a_missing_cuda_library_lowers_confidence_but_not_the_verdict() -> None:
    facts = TorchFacts(
        version="2.11.0+cu128",
        cuda_library=False,
        hip_library=False,
        version_py_found=True,
        built_cuda="12.8",
    )
    build = classify_torch(facts)
    assert (build.kind, build.confidence) == (TorchBuildKind.CUDA, Confidence.MEDIUM)
    assert build.conflicts


@pending
def test_every_signal_is_kept_as_evidence() -> None:
    build = classify_torch(CONDA_CUDA)
    keys = {(e.source, e.key) for e in build.evidence}
    assert ("torch/version.py", "cuda") in keys
    assert ("torch/lib", "cuda_library") in keys
    assert ("conda-meta", "pytorch") in keys
