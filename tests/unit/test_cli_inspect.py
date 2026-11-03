"""The inspect command, with the orchestrator replaced by a canned snapshot."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Any

import pytest

from gpu_stack_inspector import cli, orchestrator
from gpu_stack_inspector.models import (
    Distribution,
    HostFacts,
    InterpreterFacts,
    NvidiaFacts,
    Snapshot,
)
from gpu_stack_inspector.privacy import RedactionContext, strip_control

CANNED = Snapshot(
    tool_version="0.1.0",
    created_at="2026-11-03T12:00:00Z",
    host=HostFacts("Linux", "x86_64", "7.0", "Ubuntu 26.04.1 LTS"),
    nvidia=NvidiaFacts(None),
    interpreters=(
        InterpreterFacts(
            "A",
            "/home/alice/envs/a/bin/python",
            version="3.12.13",
            implementation="cpython",
            distributions=(Distribution("torch", "2.11.0+cpu", "/home/alice/envs/a/lib"),),
        ),
    ),
)


@pytest.fixture
def requested(monkeypatch: pytest.MonkeyPatch) -> list[Sequence[str]]:
    calls: list[Sequence[str]] = []

    def fake_inspect(runner: Any, pythons: Sequence[str], **_: Any) -> Snapshot:
        calls.append(list(pythons))
        return CANNED

    monkeypatch.setattr(orchestrator, "inspect", fake_inspect)
    monkeypatch.setattr(
        RedactionContext,
        "current",
        classmethod(lambda cls: RedactionContext("/home/alice", "alice", "alice-laptop")),
    )
    return calls


def run_main(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["gpu-stack-inspector", *args])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    code = exit_info.value.code
    return code if isinstance(code, int) else 0


def test_json_output_is_redacted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], requested: list[Any]
) -> None:
    assert run_main(monkeypatch, "inspect", "--python", "/x/python", "--format", "json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["interpreters"][0]["path"] == "~/envs/a/bin/python"
    assert data["interpreters"][0]["distributions"][0]["location"] == "~/envs/a/lib"
    assert requested == [["/x/python"]]


def test_text_output_is_the_terminal_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], requested: list[Any]
) -> None:
    assert run_main(monkeypatch, "inspect", "-p", "/a/python", "-p", "/b/python") == 0
    out = " ".join(strip_control(capsys.readouterr().out).split())
    assert "Python 3.12.13 (cpython) · torch 2.11.0+cpu" in out
    assert "~/envs/a/bin/python" in out
    assert "/home/alice" not in out
    assert requested == [["/a/python", "/b/python"]]


def test_an_unknown_format_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, requested: list[Any]
) -> None:
    assert run_main(monkeypatch, "inspect", "--format", "yaml") == 2
    assert requested == []


def test_inspect_is_listed_in_the_help(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_main(monkeypatch, "--help") == 0
    assert "inspect" in strip_control(capsys.readouterr().out)


def test_without_python_the_python_on_path_is_inspected(
    monkeypatch: pytest.MonkeyPatch, requested: list[Any]
) -> None:
    monkeypatch.setattr(orchestrator, "default_interpreters", lambda: ["/usr/bin/python3"])
    assert run_main(monkeypatch, "inspect", "--format", "json") == 0
    assert requested == [["/usr/bin/python3"]]
