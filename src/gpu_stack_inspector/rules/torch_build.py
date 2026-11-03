"""What kind of PyTorch build an interpreter has, from static evidence only.

No single signal is reliable: PyPI's CUDA wheels have no ``+cu`` tag, conda's builds
have neither a tag nor nvidia-* wheels, and a damaged install can have the metadata of
one build and the files of another. Each signal votes; agreement gives confidence,
disagreement between strong signals gives ``unknown`` with the conflict spelled out.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from gpu_stack_inspector.models import Confidence, Evidence, TorchFacts


class TorchBuildKind(StrEnum):
    MISSING = "missing"
    CUDA = "cuda"
    CPU = "cpu"
    ROCM = "rocm"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TorchBuild:
    """The verdict, the CUDA version the build targets when known, and why."""

    kind: TorchBuildKind
    confidence: Confidence
    cuda_version: str | None = None
    evidence: tuple[Evidence, ...] = ()
    conflicts: tuple[str, ...] = ()


def classify_torch(torch: TorchFacts | None) -> TorchBuild:
    raise NotImplementedError
