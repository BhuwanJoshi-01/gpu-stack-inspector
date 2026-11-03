"""The rules list and explain commands."""

from __future__ import annotations

import sys

import pytest

from gpu_stack_inspector import cli
from gpu_stack_inspector.privacy import strip_control
from gpu_stack_inspector.rules.catalog import builtin_rules


def run_main(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["gpu-stack-inspector", *args])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    code = exit_info.value.code
    return code if isinstance(code, int) else 0


def output(capsys: pytest.CaptureFixture[str]) -> str:
    captured = capsys.readouterr()
    return " ".join(strip_control(captured.out + captured.err).split())


def test_rules_list_shows_every_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_main(monkeypatch, "rules", "list") == 0
    text = output(capsys)
    for rule in builtin_rules():
        assert f"{rule.id} {rule.title}" in text


def test_rules_explain_prints_the_explanation_and_sources(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_main(monkeypatch, "rules", "explain", "gsi-cuda-003") == 0
    text = output(capsys)
    assert "GSI-CUDA-003 The driver supports the CUDA version PyTorch was built for" in text
    assert "cudaErrorCallRequiresNewerDriver" in text
    assert "Sources: https://docs.nvidia.com/deploy/cuda-compatibility/" in text


def test_explaining_an_unknown_code_is_an_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_main(monkeypatch, "rules", "explain", "GSI-NOPE-999") == 2
    assert "Unknown code GSI-NOPE-999" in output(capsys)


def test_rules_without_a_subcommand_shows_help(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_main(monkeypatch, "rules")
    text = output(capsys)
    assert "list" in text
    assert "explain" in text
