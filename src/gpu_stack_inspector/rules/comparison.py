"""Field-by-field comparison of inspected interpreters. Pure: facts in, rows out."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from gpu_stack_inspector.models import FieldComparison, InterpreterFacts
from gpu_stack_inspector.rules.torch_build import TorchBuildKind, classify_torch


def _flag(value: bool | None) -> str | None:
    return None if value is None else ("yes" if value else "no")


_INTERPRETER_FIELDS: tuple[tuple[str, Callable[[InterpreterFacts], str | None]], ...] = (
    ("Python", lambda i: i.version),
    ("implementation", lambda i: i.implementation),
    ("conda env", lambda i: _flag(i.conda)),
    ("venv", lambda i: _flag(i.virtualenv)),
    ("platform", lambda i: i.platform),
    ("user site", lambda i: _flag(i.user_site)),
    ("PYTHONPATH set", lambda i: _flag(i.pythonpath_set)),
)


def _torch_build(interpreter: InterpreterFacts) -> str | None:
    if interpreter.version is None:
        return None
    build = classify_torch(interpreter.torch)
    if build.kind is TorchBuildKind.CUDA:
        return f"CUDA {build.cuda_version}" if build.cuda_version else "CUDA"
    return {
        TorchBuildKind.CPU: "CPU only",
        TorchBuildKind.ROCM: "ROCm",
        TorchBuildKind.UNKNOWN: "unknown",
        TorchBuildKind.MISSING: "not installed",
    }[build.kind]


_TORCH_FIELDS: tuple[tuple[str, Callable[[InterpreterFacts], str | None]], ...] = (
    ("torch", lambda i: i.torch.version if i.torch else None),
    ("torch build", _torch_build),
    ("torch location", lambda i: i.torch.location if i.torch else None),
)


def compare_interpreters(interpreters: Sequence[InterpreterFacts]) -> tuple[FieldComparison, ...]:
    """Rows for every compared field, in a fixed order; ``differs`` marks disagreement."""
    return tuple(
        FieldComparison(name, tuple(value(i) for i in interpreters))
        for name, value in (*_INTERPRETER_FIELDS, *_TORCH_FIELDS)
    )
