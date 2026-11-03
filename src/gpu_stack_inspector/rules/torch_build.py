"""What kind of PyTorch build an interpreter has, from static evidence only.

No single signal is reliable: PyPI's CUDA wheels have no ``+cu`` tag, conda's builds
have neither a tag nor nvidia-* wheels, and a damaged install can have the metadata of
one build and the files of another. Each signal votes; agreement gives confidence,
disagreement between strong signals gives ``unknown`` with the conflict spelled out.
"""

from __future__ import annotations

import re
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


_TAG = re.compile(r"\+(?P<tag>[A-Za-z]+)(?P<number>[0-9.]*)")
_CONDA_PYTORCH = re.compile(r"pytorch-(?P<version>[0-9][^-]*)-(?P<build>.+)")
_CONDA_MUTEX = re.compile(r"pytorch-mutex-[0-9.]+-(?P<flavour>[A-Za-z]+)")
_CONDA_PYTORCH_CUDA = re.compile(r"pytorch-cuda-(?P<version>[0-9]+\.[0-9]+)-")
_CUDA_IN_BUILD = re.compile(r"cuda(?P<digits>[0-9]+)(?:\.(?P<minor>[0-9]+))?")

# Which signal's CUDA version wins when several name one: the build's own record first.
_VERSION_PY, _TAG_PRIORITY, _CONDA, _OTHER = 0, 1, 2, 9


@dataclass(frozen=True, slots=True)
class _Vote:
    kind: TorchBuildKind
    strong: bool
    source: str
    """Human description, used in conflict messages: "the version tag +cu128"."""
    cuda_version: str | None = None
    priority: int = _OTHER


def classify_torch(torch: TorchFacts | None) -> TorchBuild:
    """Classify the build from the evidence of the torch an interpreter would import."""
    if torch is None:
        return TorchBuild(
            TorchBuildKind.MISSING,
            Confidence.HIGH,
            evidence=(Evidence("python-metadata", "torch", None),),
        )
    votes: list[_Vote] = []
    evidence: list[Evidence] = [Evidence("torch-metadata", "version", torch.version)]
    _version_tag(torch, votes)
    _version_py(torch, votes, evidence)
    _libraries(torch, votes, evidence)
    _requirements(torch, votes, evidence)
    _conda(torch, votes, evidence)
    return _decide(votes, tuple(evidence))


def cuda_from_digits(digits: str) -> str:
    """``128`` → ``12.8``, ``92`` → ``9.2``: how PyTorch writes CUDA versions in tags."""
    return f"{digits[:-1]}.{digits[-1]}" if len(digits) >= 2 else digits


def _version_tag(torch: TorchFacts, votes: list[_Vote]) -> None:
    match = _TAG.search(torch.version)
    if match is None:
        return  # PyPI's CUDA wheels and conda builds have no tag
    tag, number = match["tag"].lower(), match["number"]
    source = f"the version tag +{tag}{number}"
    if tag == "cu" and number.isdigit():
        votes.append(
            _Vote(TorchBuildKind.CUDA, True, source, cuda_from_digits(number), _TAG_PRIORITY)
        )
    elif tag == "cpu":
        votes.append(_Vote(TorchBuildKind.CPU, True, source))
    elif tag == "rocm":
        votes.append(_Vote(TorchBuildKind.ROCM, True, source))


def _version_py(torch: TorchFacts, votes: list[_Vote], evidence: list[Evidence]) -> None:
    if not torch.version_py_found:
        evidence.append(Evidence("torch/version.py", "found", "no"))
        return
    evidence.append(Evidence("torch/version.py", "cuda", torch.built_cuda or "None"))
    evidence.append(Evidence("torch/version.py", "hip", torch.built_hip or "None"))
    if torch.built_hip:
        votes.append(_Vote(TorchBuildKind.ROCM, True, f"torch/version.py (hip {torch.built_hip})"))
    elif torch.built_cuda:
        source = f"torch/version.py (cuda {torch.built_cuda})"
        votes.append(_Vote(TorchBuildKind.CUDA, True, source, torch.built_cuda, _VERSION_PY))
    else:
        votes.append(_Vote(TorchBuildKind.CPU, True, "torch/version.py (cuda None)"))


def _libraries(torch: TorchFacts, votes: list[_Vote], evidence: list[Evidence]) -> None:
    for key, present in (("cuda_library", torch.cuda_library), ("hip_library", torch.hip_library)):
        if present is not None:
            evidence.append(Evidence("torch/lib", key, "present" if present else "absent"))
    if torch.cuda_library:
        votes.append(_Vote(TorchBuildKind.CUDA, True, "libtorch_cuda in torch/lib"))
    elif torch.hip_library:
        votes.append(_Vote(TorchBuildKind.ROCM, True, "libtorch_hip in torch/lib"))
    elif torch.cuda_library is False and torch.hip_library is False:
        # Weak: files can be missing from a damaged install of a GPU build.
        votes.append(_Vote(TorchBuildKind.CPU, False, "no GPU library in torch/lib"))


def _requirements(torch: TorchFacts, votes: list[_Vote], evidence: list[Evidence]) -> None:
    if torch.requires_nvidia:
        evidence.append(
            Evidence("torch-metadata", "requires_nvidia", ", ".join(torch.requires_nvidia))
        )
        votes.append(_Vote(TorchBuildKind.CUDA, False, "its nvidia-* requirements"))


def _conda(torch: TorchFacts, votes: list[_Vote], evidence: list[Evidence]) -> None:
    for record in torch.conda_records or ():
        if (match := _CONDA_PYTORCH.fullmatch(record)) is not None:
            evidence.append(Evidence("conda-meta", "pytorch", record))
            build = match["build"].lower()
            source = f"conda's pytorch build {match['build']}"
            if (cuda := _CUDA_IN_BUILD.search(build)) is not None:
                version = (
                    f"{cuda['digits']}.{cuda['minor']}"
                    if cuda["minor"] is not None
                    else cuda_from_digits(cuda["digits"])
                )
                votes.append(_Vote(TorchBuildKind.CUDA, True, source, version, _CONDA))
            elif "cpu" in build:
                votes.append(_Vote(TorchBuildKind.CPU, True, source))
        elif (mutex := _CONDA_MUTEX.fullmatch(record)) is not None:
            evidence.append(Evidence("conda-meta", "pytorch-mutex", record))
            flavour = mutex["flavour"].lower()
            kind = {"cuda": TorchBuildKind.CUDA, "cpu": TorchBuildKind.CPU}.get(flavour)
            if kind is not None:
                votes.append(_Vote(kind, False, f"conda's pytorch-mutex ({flavour})"))
        elif (pytorch_cuda := _CONDA_PYTORCH_CUDA.match(record)) is not None:
            evidence.append(Evidence("conda-meta", "pytorch-cuda", record))
            votes.append(
                _Vote(
                    TorchBuildKind.CUDA,
                    False,
                    "conda's pytorch-cuda",
                    pytorch_cuda["version"],
                    _CONDA,
                )
            )


_LOWER = {
    Confidence.HIGH: Confidence.MEDIUM,
    Confidence.MEDIUM: Confidence.LOW,
    Confidence.LOW: Confidence.LOW,
}


def _decide(votes: list[_Vote], evidence: tuple[Evidence, ...]) -> TorchBuild:
    if not votes:
        return TorchBuild(TorchBuildKind.UNKNOWN, Confidence.LOW, evidence=evidence)
    strong_kinds = {vote.kind for vote in votes if vote.strong}
    if len(strong_kinds) > 1:
        return TorchBuild(
            TorchBuildKind.UNKNOWN,
            Confidence.LOW,
            evidence=evidence,
            conflicts=tuple(_conflicts(votes)),
        )
    if strong_kinds:
        kind = next(iter(strong_kinds))
        agreeing_strong = sum(vote.strong for vote in votes)
        confidence = Confidence.HIGH if agreeing_strong >= 2 else Confidence.MEDIUM
    else:
        weak_kinds = {vote.kind for vote in votes}
        if len(weak_kinds) > 1:
            return TorchBuild(
                TorchBuildKind.UNKNOWN,
                Confidence.LOW,
                evidence=evidence,
                conflicts=tuple(_conflicts(votes)),
            )
        kind = next(iter(weak_kinds))
        confidence = Confidence.MEDIUM if len(votes) >= 2 else Confidence.LOW

    conflicts = _conflicts(votes) if any(vote.kind is not kind for vote in votes) else []
    cuda_version = None
    if kind is TorchBuildKind.CUDA:
        named = sorted((v for v in votes if v.cuda_version), key=lambda v: v.priority)
        if named:
            cuda_version = named[0].cuda_version
            for other in named[1:]:
                if other.cuda_version != cuda_version:
                    conflicts.append(
                        f"{named[0].source} says CUDA {cuda_version}, "
                        f"but {other.source} says CUDA {other.cuda_version}"
                    )
    if conflicts:
        confidence = _LOWER[confidence]
    return TorchBuild(kind, confidence, cuda_version, evidence, tuple(conflicts))


def _conflicts(votes: list[_Vote]) -> list[str]:
    """One sentence per pair of disagreeing kinds, naming their strongest witnesses."""
    witnesses: dict[TorchBuildKind, _Vote] = {}
    for vote in sorted(votes, key=lambda v: not v.strong):
        witnesses.setdefault(vote.kind, vote)
    kinds = list(witnesses)
    return [
        f"{witnesses[a].source} says {a.value}, but {witnesses[b].source} says {b.value}"
        for i, a in enumerate(kinds)
        for b in kinds[i + 1 :]
    ]
