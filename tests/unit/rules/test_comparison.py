"""Comparing interpreters field by field."""

from __future__ import annotations

from gpu_stack_inspector.models import FieldComparison, InterpreterFacts
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
