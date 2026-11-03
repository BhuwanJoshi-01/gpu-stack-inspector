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


def torch_build_label(interpreter: InterpreterFacts) -> str | None:
    """The build in plain words ("CUDA 12.8", "CPU only", "not installed"…)."""
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
    ("torch build", torch_build_label),
    ("torch location", lambda i: i.torch.location if i.torch else None),
)


def _package_version(interpreter: InterpreterFacts, name: str) -> str | None:
    """The installed version; several installs of one package (shadowing) are all named."""
    versions = [d.version for d in interpreter.distributions if d.name == name]
    if not versions:
        return None
    if len(versions) == 1:
        return versions[0]
    return f"{', '.join(sorted(set(versions)))} ({len(versions)} installs)"


def compare_interpreters(interpreters: Sequence[InterpreterFacts]) -> tuple[FieldComparison, ...]:
    """Rows for every compared field, in a fixed order; ``differs`` marks disagreement.

    After the interpreter and torch rows comes one row per GPU-relevant package found in
    any of them, alphabetically; ``None`` there means "not installed".
    """
    fixed = tuple(
        FieldComparison(name, tuple(value(i) for i in interpreters))
        for name, value in (*_INTERPRETER_FIELDS, *_TORCH_FIELDS)
    )
    names = sorted({d.name for i in interpreters for d in i.distributions} - {"torch"})
    packages = tuple(
        FieldComparison(name, tuple(_package_version(i, name) for i in interpreters))
        for name in names
    )
    return fixed + packages
