"""The Markdown dossier."""

from __future__ import annotations

import dataclasses

from gpu_stack_inspector.models import (
    Confidence,
    Finding,
    InterpreterFacts,
    RedactionCount,
    Severity,
    Snapshot,
    Status,
)
from gpu_stack_inspector.reports.markdown import code, render_markdown, text
from tests.contract.test_json_schema import real_laptop_snapshot


def test_text_escapes_markdown_and_html() -> None:
    assert text("a|b *c* _d_ [e](f) <script> #1 `x`") == (
        r"a\|b \*c\* \_d\_ \[e\](f) \<script\> \#1 \`x\`"
    )
    assert text("two\nlines") == "two lines"


def test_code_spans_survive_backticks() -> None:
    assert code("~/envs/a/bin/python") == "`~/envs/a/bin/python`"
    assert code("echo `id`") == "`` echo `id` ``"
    assert code("a``b") == "```a``b```"


def test_the_real_laptop_report_reads_as_a_dossier() -> None:
    report = render_markdown(
        dataclasses.replace(real_laptop_snapshot(), redactions=(RedactionCount("home_path", 9),))
    )
    assert report.startswith("# GPU Stack Inspector report\n")
    assert "Read-only: nothing was installed or changed." in report
    assert "Redacted from this output: 9 home paths." in report
    assert "## Summary" in report
    assert "- **FAIL** `GSI-TORCH-002` (B): torch 2.15.0+cpu is a CPU-only build" in report
    assert "| NVIDIA driver | 595.99.02 (supports CUDA up to 13.2) |" in report
    assert "| GPU 0 | NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, compute 8.6 |" in report
    assert "## Environments" in report
    assert "## Differences" in report
    assert "| torch build | CUDA 12.8 | CPU only | not installed |" in report
    assert "### Host" in report
    assert "  - Check: `gpu-stack-inspector inspect --python ~/cpu-venv/bin/python`" in report


def test_a_path_with_a_pipe_does_not_break_the_tables() -> None:
    snapshot = Snapshot(
        "0.1.0", "t", interpreters=(InterpreterFacts("A", "/envs/a|b/python", version="3.12.0"),)
    )
    row = next(line for line in render_markdown(snapshot).splitlines() if line.startswith("| A |"))
    assert row.count(" | ") == 6  # seven cells, the pipe inside the code span does not split one


def test_no_problems_and_no_checks_are_said_plainly() -> None:
    passing = Finding("GSI-SYS-001", Status.PASS, Severity.INFO, Confidence.HIGH, "ok", "fine")
    assert "No problems found." in render_markdown(Snapshot("0.1.0", "t", findings=(passing,)))
    assert "No checks were run." in render_markdown(Snapshot("0.1.0", "t"))


def test_consequences_are_not_listed_as_separate_problems() -> None:
    root = Finding(
        "GSI-SYS-001",
        Status.WARN,
        Severity.MEDIUM,
        Confidence.MEDIUM,
        "nvidia-smi was not found",
        "r",
    )
    follow = Finding(
        "GSI-SYS-002",
        Status.UNKNOWN,
        Severity.INFO,
        Confidence.HIGH,
        "driver unknown",
        "r",
        caused_by="GSI-SYS-001",
    )
    report = render_markdown(Snapshot("0.1.0", "t", findings=(root, follow)))
    summary = report.split("## Summary")[1].split("##")[0]
    assert "GSI-SYS-001" in summary
    assert "GSI-SYS-002" not in summary
    assert "(follows from `GSI-SYS-001`)" in report


def test_the_environments_table_says_whether_cuda_was_tested() -> None:
    env = InterpreterFacts("A", "/a/python", version="3.12.0")
    row = next(
        line
        for line in render_markdown(Snapshot("0.1.0", "t", interpreters=(env,))).splitlines()
        if line.startswith("| A |")
    )
    assert row.endswith("| not tested |")
