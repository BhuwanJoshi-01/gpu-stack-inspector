"""The probe command: consent first, then the runtime check."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest

from gpu_stack_inspector import cli, orchestrator
from gpu_stack_inspector.models import InterpreterFacts, Snapshot, TorchFacts
from gpu_stack_inspector.privacy import RedactionContext, strip_control
from gpu_stack_inspector.probes.runtime_probe import parse_runtime_output
from tests.fakes import read_fixture

ENV = InterpreterFacts(
    "A", "/envs/a/bin/python", version="3.13.13",
    torch=TorchFacts("2.11.0+cu128", cuda_library=True, version_py_found=True, built_cuda="12.8"),
)  # fmt: skip


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def fake_inspect(runner: Any, pythons: Sequence[str], **kwargs: Any) -> Snapshot:
        seen.append({"pythons": list(pythons), **kwargs})
        env = ENV
        if kwargs.get("runtime"):
            env = replace(
                ENV, runtime=parse_runtime_output(read_fixture("runtime", "sapiens2.stdout.txt"))
            )
        return Snapshot("0.1.0", "t", interpreters=(env,))

    monkeypatch.setattr(orchestrator, "inspect", fake_inspect)
    monkeypatch.setattr(orchestrator, "default_interpreters", lambda: ["/usr/bin/python3"])
    monkeypatch.setattr(
        RedactionContext, "current", classmethod(lambda cls: RedactionContext(None, None, None))
    )
    return seen


def run_main(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["gpu-stack-inspector", *args])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    code = exit_info.value.code
    return code if isinstance(code, int) else 0


def test_without_runtime_nothing_runs_and_the_consent_is_explained(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], calls: list[Any]
) -> None:
    assert run_main(monkeypatch, "probe", "-p", "/envs/a/bin/python") == 2
    assert calls == []
    err = " ".join(capsys.readouterr().err.split())
    assert "uses a little GPU memory" in err
    assert "--runtime to agree" in err


def test_with_runtime_the_check_runs_and_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], calls: list[Any]
) -> None:
    assert run_main(monkeypatch, "probe", "--runtime", "-p", "/envs/a/bin/python") == 0
    assert calls[0]["runtime"] is True
    captured = capsys.readouterr()
    assert "Importing torch and running a tiny CUDA operation in 1 interpreter(s)" in captured.err
    out = " ".join(strip_control(captured.out).split())
    assert "PASS GSI-TORCH-003 A CUDA operation ran on NVIDIA GeForce RTX 3050 Laptop GPU" in out
    assert "GSI-TORCH-004" not in out


def test_the_runtime_result_is_in_the_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], calls: list[Any]
) -> None:
    assert run_main(monkeypatch, "probe", "--runtime", "-p", "/a", "--format", "json") == 0
    runtime = json.loads(capsys.readouterr().out)["interpreters"][0]["runtime"]
    assert runtime["tensor_ok"] is True
    assert runtime["devices"][0]["compute_capability"] == "8.6"


def test_inspect_never_runs_the_runtime_check(
    monkeypatch: pytest.MonkeyPatch, calls: list[Any]
) -> None:
    run_main(monkeypatch, "inspect", "-p", "/a", "--format", "json")
    assert calls[0].get("runtime", False) is False
