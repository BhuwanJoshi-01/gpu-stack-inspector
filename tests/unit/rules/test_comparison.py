"""Comparing interpreters field by field."""

from __future__ import annotations

from dataclasses import replace

from gpu_stack_inspector.models import FieldComparison, InterpreterFacts, TorchFacts
from gpu_stack_inspector.rules.comparison import compare_interpreters

CONDA = InterpreterFacts(
    "A", "/envs/a/bin/python", version="3.13.13", implementation="cpython", conda=True,
    prefix="/envs/a", base_prefix="/envs/a", platform="linux-x86_64", user_site=False,
    pythonpath_set=False,
)  # fmt: skip
VENV = InterpreterFacts(
    "B", "/venv/bin/python", version="3.12.13", implementation="cpython", conda=False,
    prefix="/venv", base_prefix="/usr", platform="linux-x86_64", user_site=False,
    pythonpath_set=True,
)  # fmt: skip


def rows(*interpreters: InterpreterFacts) -> dict[str, FieldComparison]:
    return {row.field: row for row in compare_interpreters(interpreters)}


def test_interpreter_fields_are_compared_in_order() -> None:
    result = rows(CONDA, VENV)
    assert result["Python"].values == ("3.13.13", "3.12.13")
    assert result["Python"].differs
    assert result["conda env"].values == ("yes", "no")
    assert result["venv"].values == ("no", "yes")
    assert result["PYTHONPATH set"].differs
    assert not result["platform"].differs
    assert not result["implementation"].differs


def test_unknown_values_are_none_and_count_as_different() -> None:
    unusable = InterpreterFacts("C", "/gone/python")
    result = rows(CONDA, unusable)
    assert result["Python"].values == ("3.13.13", None)
    assert result["Python"].differs
    assert result["venv"].values == ("no", None)


def test_a_single_interpreter_never_differs() -> None:
    assert not any(row.differs for row in compare_interpreters([CONDA]))


CUDA_TORCH = TorchFacts(
    "2.11.0+cu128", "/envs/a/lib/site-packages", cuda_library=True, version_py_found=True,
    built_cuda="12.8",
)  # fmt: skip
CPU_TORCH = TorchFacts(
    "2.15.0+cpu", "/venv/lib/site-packages", cuda_library=False, hip_library=False,
    version_py_found=True,
)  # fmt: skip


def test_the_torch_build_is_the_row_that_explains_the_difference() -> None:
    result = rows(replace(CONDA, torch=CUDA_TORCH), replace(VENV, torch=CPU_TORCH))
    assert result["torch"].values == ("2.11.0+cu128", "2.15.0+cpu")
    assert result["torch build"].values == ("CUDA 12.8", "CPU only")
    assert result["torch location"].values == (
        "/envs/a/lib/site-packages",
        "/venv/lib/site-packages",
    )


def test_an_environment_without_torch_says_not_installed() -> None:
    result = rows(replace(CONDA, torch=CUDA_TORCH), VENV)
    assert result["torch"].values == ("2.11.0+cu128", None)
    assert result["torch build"].values == ("CUDA 12.8", "not installed")
    assert result["torch build"].differs
