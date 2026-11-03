"""The runtime probe program, run against fake torch packages."""

from __future__ import annotations

import ast
import json
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

from gpu_stack_inspector.runner import Outcome, SubprocessRunner
from tests.fakes import FIXTURES

SOURCE = files("gpu_stack_inspector.probes").joinpath("runtime_program.py").read_text()
ALLOWED_IMPORTS = {"__future__", "json", "sys", "time", "typing", "torch"}


def run(case: str) -> dict[str, Any]:
    fake = FIXTURES / "fake-torch" / case
    result = SubprocessRunner().run(
        [sys.executable, "-c", SOURCE], timeout=30, env_overrides={"PYTHONPATH": str(fake)}
    )
    assert result.outcome is Outcome.OK, result.stderr
    start = result.stdout.index("GSI-RUNTIME-BEGIN") + len("GSI-RUNTIME-BEGIN")
    document: dict[str, Any] = json.loads(
        result.stdout[start : result.stdout.index("GSI-RUNTIME-END")]
    )
    return document


def steps(document: dict[str, Any]) -> list[tuple[str, bool]]:
    return [(step["step"], step["ok"]) for step in document["steps"]]


def test_the_program_is_valid_python_3_8() -> None:
    ast.parse(SOURCE, feature_version=(3, 8))


def test_the_program_imports_only_the_standard_library_and_torch() -> None:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported <= ALLOWED_IMPORTS, imported - ALLOWED_IMPORTS


def test_a_working_cuda_install_runs_every_step() -> None:
    document = run("working")
    assert document["protocol"] == 1
    assert steps(document) == [
        ("import torch", True),
        ("cuda available", True),
        ("devices", True),
        ("tensor operation", True),
    ]
    by_name = {step["step"]: step for step in document["steps"]}
    assert by_name["import torch"]["value"] == {"version": "2.11.0+cu128", "cuda": "12.8"}
    assert by_name["devices"]["value"] == [
        {"index": 0, "name": "NVIDIA GeForce RTX 3050 Laptop GPU", "capability": "8.6"}
    ]
    assert by_name["tensor operation"]["value"] == 32.0


def test_a_cpu_only_build_stops_after_cuda_available() -> None:
    document = run("cpu-only")
    assert steps(document) == [("import torch", True), ("cuda available", True)]
    assert document["steps"][1]["value"] is False


def test_a_missing_kernel_image_fails_the_tensor_operation_with_the_real_message() -> None:
    document = run("no-kernel-image")
    assert steps(document)[-1] == ("tensor operation", False)
    assert document["steps"][-1]["error"] == (
        "RuntimeError: CUDA error: no kernel image is available for execution on the device"
    )


def test_a_failing_import_is_reported_as_the_first_step() -> None:
    document = run("import-error")
    assert steps(document) == [("import torch", False)]
    assert "libcudnn.so.9" in document["steps"][0]["error"]


@pytest.mark.parametrize("case", ["working", "cpu-only", "no-kernel-image", "import-error"])
def test_every_step_is_timed(case: str) -> None:
    for step in run(case)["steps"]:
        assert step["seconds"] >= 0


def test_nothing_is_written_next_to_the_probe(tmp_path: Path) -> None:
    fake = FIXTURES / "fake-torch" / "working"
    SubprocessRunner().run(
        [sys.executable, "-c", SOURCE],
        timeout=30,
        cwd=tmp_path,
        env_overrides={"PYTHONPATH": str(fake)},
    )
    assert list(tmp_path.iterdir()) == []


def test_each_step_is_announced_before_it_starts() -> None:
    fake = FIXTURES / "fake-torch" / "working"
    result = SubprocessRunner().run(
        [sys.executable, "-c", SOURCE], timeout=30, env_overrides={"PYTHONPATH": str(fake)}
    )
    announced = [
        line.split(" ", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("GSI-RUNTIME-STEP ")
    ]
    assert announced == ["import torch", "cuda available", "devices", "tensor operation"]
    assert result.stdout.index("GSI-RUNTIME-STEP tensor operation") < result.stdout.index(
        "GSI-RUNTIME-BEGIN"
    )
