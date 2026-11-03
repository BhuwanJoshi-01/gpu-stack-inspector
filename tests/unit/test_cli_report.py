"""The report command, with canned facts."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from gpu_stack_inspector import cli, orchestrator
from gpu_stack_inspector.models import InterpreterFacts, Snapshot, TorchFacts
from gpu_stack_inspector.privacy import RedactionContext

WORKS = InterpreterFacts(
    "A", "/home/alice/envs/works/bin/python", version="3.13.13",
    torch=TorchFacts("2.11.0+cu128", cuda_library=True, version_py_found=True, built_cuda="12.8"),
)  # fmt: skip
BROKEN = InterpreterFacts(
    "B", "/home/alice/envs/broken/bin/python", version="3.12.13",
    torch=TorchFacts("2.15.0+cpu", cuda_library=False, hip_library=False, version_py_found=True),
)  # fmt: skip


@pytest.fixture(autouse=True)
def canned(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_inspect(runner: Any, pythons: Sequence[str], **_: Any) -> Snapshot:
        return Snapshot(
            "0.1.0", "2026-11-03T12:00:00Z", interpreters=(WORKS, BROKEN)[: len(pythons)]
        )

    monkeypatch.setattr(orchestrator, "inspect", fake_inspect)
    monkeypatch.setattr(orchestrator, "default_interpreters", lambda: ["/usr/bin/python3"])
    monkeypatch.setattr(
        RedactionContext,
        "current",
        classmethod(lambda cls: RedactionContext("/home/alice", "alice", None)),
    )


def run_main(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["gpu-stack-inspector", *args])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    code = exit_info.value.code
    return code if isinstance(code, int) else 0


def test_the_markdown_report_is_printed_redacted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_main(monkeypatch, "report", "-p", "/a") == 0
    out = capsys.readouterr().out
    assert out.startswith("# GPU Stack Inspector report\n")
    assert "`~/envs/works/bin/python`" in out
    assert "/home/alice" not in out
    assert "Redacted from this output:" in out
    assert "home path" in out
    assert "## Differences" not in out


def test_two_interpreters_bring_their_comparison(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_main(monkeypatch, "report", "-p", "/a", "-p", "/b") == 0
    out = capsys.readouterr().out
    assert "| torch build | CUDA 12.8 | CPU only |" in out
    assert "GSI-DIFF-001" in out


def test_a_json_report_matches_the_inspect_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_main(monkeypatch, "report", "-p", "/a", "-p", "/b", "--format", "json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema_version"] == 1
    assert data["comparison"]
    assert [item["kind"] for item in data["redactions"]] == ["home_path"]
    assert "/home/alice" not in json.dumps(data)


def test_output_writes_only_the_named_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    target = tmp_path / "gpu-report.md"
    assert run_main(monkeypatch, "report", "-p", "/a", "--output", str(target)) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["gpu-report.md"]
    assert target.read_text(encoding="utf-8").startswith("# GPU Stack Inspector report")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"Wrote {target}." in captured.err
    assert "Read it before you share it." in captured.err


def test_an_existing_file_is_not_overwritten_without_force(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    target = tmp_path / "gpu-report.md"
    target.write_text("keep me", encoding="utf-8")
    assert run_main(monkeypatch, "report", "-p", "/a", "-o", str(target)) == 2
    assert target.read_text(encoding="utf-8") == "keep me"
    assert "use --force" in capsys.readouterr().err
    assert run_main(monkeypatch, "report", "-p", "/a", "-o", str(target), "--force") == 0
    assert target.read_text(encoding="utf-8").startswith("# GPU Stack Inspector report")


def test_a_dash_prints_the_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_main(monkeypatch, "report", "-p", "/a", "--output", "-") == 0
    assert capsys.readouterr().out.startswith("# GPU Stack Inspector report")
