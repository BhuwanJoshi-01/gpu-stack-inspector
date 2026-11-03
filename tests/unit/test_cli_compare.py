"""The compare command, with the orchestrator's inspection replaced by canned facts."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Any

import pytest

from gpu_stack_inspector import cli, orchestrator
from gpu_stack_inspector.models import (
    FailureKind,
    InterpreterFacts,
    ProbeFailure,
    Snapshot,
    TorchFacts,
)
from gpu_stack_inspector.privacy import RedactionContext, strip_control

WORKS = InterpreterFacts(
    "A", "/home/alice/envs/works/bin/python", version="3.13.13", implementation="cpython",
    torch=TorchFacts("2.11.0+cu128", cuda_library=True, version_py_found=True, built_cuda="12.8"),
)  # fmt: skip
BROKEN = InterpreterFacts(
    "B", "/home/alice/envs/broken/bin/python", version="3.12.13", implementation="cpython",
    torch=TorchFacts("2.15.0+cpu", cuda_library=False, hip_library=False, version_py_found=True),
)  # fmt: skip


@pytest.fixture
def interpreters(monkeypatch: pytest.MonkeyPatch) -> list[InterpreterFacts]:
    chosen = [WORKS, BROKEN]

    def fake_inspect(runner: Any, pythons: Sequence[str], **_: Any) -> Snapshot:
        return Snapshot("0.1.0", "t", interpreters=tuple(chosen[: len(pythons)]))

    monkeypatch.setattr(orchestrator, "inspect", fake_inspect)
    monkeypatch.setattr(orchestrator, "default_interpreters", lambda: ["/usr/bin/python3"])
    monkeypatch.setattr(
        RedactionContext,
        "current",
        classmethod(lambda cls: RedactionContext("/home/alice", "alice", None)),
    )
    return chosen


def run_main(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["gpu-stack-inspector", *args])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    code = exit_info.value.code
    return code if isinstance(code, int) else 0


def text(capsys: pytest.CaptureFixture[str]) -> list[str]:
    captured = capsys.readouterr()
    return [
        " ".join(line.split()) for line in strip_control(captured.out + captured.err).splitlines()
    ]


def test_two_environments_are_compared_with_the_difference_explained(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], interpreters: list[Any]
) -> None:
    assert run_main(monkeypatch, "compare", "-p", "/a", "-p", "/b") == 0
    lines = text(capsys)
    joined = " ".join(lines)  # long findings wrap at the 80 columns of a captured stream
    assert "A ~/envs/works/bin/python" in lines
    assert "torch build CUDA 12.8 CPU only" in lines
    assert (
        "GSI-DIFF-001 The environments have different PyTorch builds: A CUDA 12.8, B CPU only"
        in joined
    )
    assert "FAIL GSI-TORCH-002 torch 2.15.0+cpu is a CPU-only build" in joined


def test_json_output_has_the_comparison_rows(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], interpreters: list[Any]
) -> None:
    assert run_main(monkeypatch, "compare", "-p", "/a", "-p", "/b", "--format", "json") == 0
    data = json.loads(capsys.readouterr().out)
    rows = {row["field"]: row for row in data["comparison"]}
    assert rows["torch build"] == {
        "field": "torch build",
        "values": ["CUDA 12.8", "CPU only"],
        "differs": True,
    }
    assert data["interpreters"][0]["path"] == "~/envs/works/bin/python"


def test_all_shows_identical_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], interpreters: list[Any]
) -> None:
    assert run_main(monkeypatch, "compare", "-p", "/a", "-p", "/b", "--all") == 0
    assert "implementation cpython cpython" in text(capsys)


@pytest.mark.parametrize("args", [(), ("-p", "/a")])
def test_fewer_than_two_interpreters_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    interpreters: list[Any],
    args: tuple[str, ...],
) -> None:
    assert run_main(monkeypatch, "compare", *args) == 2


def test_an_interpreter_that_cannot_be_started_exits_3(
    monkeypatch: pytest.MonkeyPatch, interpreters: list[Any]
) -> None:
    gone = ProbeFailure("python", FailureKind.NOT_FOUND, "/gone does not exist")
    interpreters[1] = InterpreterFacts("B", "/gone", failures=(gone,))
    assert run_main(monkeypatch, "compare", "-p", "/a", "-p", "/gone", "--format", "json") == 3
