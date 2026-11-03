"""Command-line interface: parses arguments and wires the layers together.

No diagnosis logic lives here; commands call collectors, rules and renderers.
"""

from __future__ import annotations

import os
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from gpu_stack_inspector import __version__, orchestrator
from gpu_stack_inspector.exit_codes import ExitCode
from gpu_stack_inspector.models import FailureKind, Snapshot
from gpu_stack_inspector.privacy import RedactionContext, Redactor, redact_snapshot
from gpu_stack_inspector.reports.json_report import render_json
from gpu_stack_inspector.reports.markdown import render_markdown
from gpu_stack_inspector.reports.terminal import redaction_line, render_compare, render_terminal
from gpu_stack_inspector.rules.catalog import builtin_rules
from gpu_stack_inspector.rules.engine import RuleContext
from gpu_stack_inspector.runner import SubprocessRunner

HELP = (
    "Explain why PyTorch can use the GPU in one Python environment but not another.\n\n"
    "Inspection is read-only: nothing is installed or changed, no root is needed and no "
    "network connection is made. PyTorch is only imported with `probe --runtime`."
)

app = typer.Typer(
    name="gpu-stack-inspector",
    help=HELP,
    no_args_is_help=True,
    add_completion=False,
    # Rich tracebacks print local variables, which can hold environment values.
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"gpu-stack-inspector {__version__}")
        raise typer.Exit()


@app.callback()
def cli(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Global options (the help text comes from HELP)."""


rules_app = typer.Typer(
    help="List the checks and explain what each one looks at.", no_args_is_help=True
)
app.add_typer(rules_app, name="rules")


@rules_app.command("list")
def rules_list() -> None:
    """List every check with its code."""
    table = Table(box=None, pad_edge=False, show_edge=False)
    table.add_column("Code", no_wrap=True)
    table.add_column("Check")
    for rule in builtin_rules():
        table.add_row(Text(rule.id), Text(rule.title))
    Console().print(table)


@rules_app.command("explain")
def rules_explain(
    code: Annotated[
        str, typer.Argument(metavar="CODE", help="A finding code, e.g. GSI-TORCH-002.")
    ],
) -> None:
    """Explain what a check looks at, how it decides, and its sources."""
    rule = builtin_rules().get(code.strip())
    if rule is None:
        typer.echo(
            f"Unknown code {code}; `gpu-stack-inspector rules list` shows them all.", err=True
        )
        raise typer.Exit(ExitCode.ERROR)
    console = Console()
    console.print(Text(f"{rule.id}  {rule.title}", style="bold"))
    console.print()
    console.print(Text(rule.explanation))
    if rule.sources:
        console.print()
        console.print(Text("Sources:", style="bold"))
        for source in rule.sources:
            console.print(Text(f"  {source}"))


class OutputFormat(StrEnum):
    TEXT = "text"
    JSON = "json"


@app.command()
def inspect(
    python: Annotated[
        list[str] | None,
        typer.Option(
            "--python",
            "-p",
            metavar="PATH",
            help=(
                "Python interpreter to inspect. Repeat it to inspect several. "
                "Default: the python on PATH."
            ),
        ),
    ] = None,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", "-f", help="Output format.")
    ] = OutputFormat.TEXT,
) -> None:
    """Inspect this machine and Python interpreters, read-only.

    Interpreters are inspected from their metadata only; nothing is imported into
    them beyond the standard library.
    """
    snapshot = _collect(python, compare=False)
    redacted = redact_snapshot(snapshot)
    if output_format is OutputFormat.JSON:
        typer.echo(render_json(redacted), nl=False)
    else:
        render_terminal(redacted, Console())
    if _unusable_interpreters(snapshot):
        raise typer.Exit(ExitCode.CONTEXT_UNAVAILABLE)


@app.command()
def compare(
    python: Annotated[
        list[str],
        typer.Option(
            "--python",
            "-p",
            metavar="PATH",
            help="An interpreter to compare. Give it at least twice.",
        ),
    ],
    output_format: Annotated[
        OutputFormat, typer.Option("--format", "-f", help="Output format.")
    ] = OutputFormat.TEXT,
    show_all: Annotated[
        bool, typer.Option("--all", help="Show the fields that are the same too.")
    ] = False,
) -> None:
    """Compare interpreters side by side: which fields differ, and what that means.

    Read-only, like inspect: interpreters are inspected from their metadata only.
    """
    if len(python) < 2:
        raise typer.BadParameter("give --python at least twice", param_hint="--python")
    snapshot = _collect(python, compare=True)
    redacted = redact_snapshot(snapshot)
    if output_format is OutputFormat.JSON:
        typer.echo(render_json(redacted), nl=False)
    else:
        render_compare(redacted, Console(), show_all=show_all)
    if _unusable_interpreters(snapshot):
        raise typer.Exit(ExitCode.CONTEXT_UNAVAILABLE)


class ReportFormat(StrEnum):
    MARKDOWN = "markdown"
    JSON = "json"


@app.command()
def report(
    python: Annotated[
        list[str] | None,
        typer.Option(
            "--python",
            "-p",
            metavar="PATH",
            help=(
                "Interpreter to include. Repeat it to compare several. Default: the python on PATH."
            ),
        ),
    ] = None,
    report_format: Annotated[
        ReportFormat, typer.Option("--format", "-f", help="Report format.")
    ] = ReportFormat.MARKDOWN,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            metavar="FILE",
            help="Write the report to FILE instead of printing it ('-' prints).",
        ),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite FILE if it exists.")] = False,
) -> None:
    """Write a redacted report to paste into a bug tracker.

    By default the report is printed, so it can be read before it is shared. With two or
    more interpreters it includes their comparison. The file named with --output is the
    only thing this tool ever writes.
    """
    to_file = output is not None and str(output) != "-"
    if to_file and output is not None and output.exists() and not force:
        typer.echo(f"{output} exists; use --force to overwrite it.", err=True)
        raise typer.Exit(ExitCode.ERROR)
    snapshot = _collect(python, compare=len(python or []) >= 2)
    redacted = redact_snapshot(snapshot)
    if report_format is ReportFormat.JSON:
        rendered = render_json(redacted)
    else:
        rendered = render_markdown(redacted)
    if to_file and output is not None:
        output.write_text(rendered, encoding="utf-8")
        note = redaction_line(redacted) or "Nothing needed redacting"
        typer.echo(f"Wrote {output}. {note}. Read it before you share it.", err=True)
    else:
        typer.echo(rendered, nl=False)
    if _unusable_interpreters(snapshot):
        raise typer.Exit(ExitCode.CONTEXT_UNAVAILABLE)


def _collect(python: list[str] | None, *, compare: bool) -> Snapshot:
    """Inspect, run every check and, if asked, compare: the shared part of each command.

    Without interpreters the python on PATH is used; the shell's python also goes into the
    rule context.
    """
    shell_python = next(iter(orchestrator.default_interpreters()), None)
    pythons = python or ([shell_python] if shell_python else [])
    if not pythons:
        typer.echo("No python found on PATH; name interpreters with --python.", err=True)
    snapshot = orchestrator.diagnose(
        orchestrator.inspect(SubprocessRunner(), pythons),
        RuleContext(shell_python=shell_python),
    )
    return orchestrator.with_comparison(snapshot) if compare else snapshot


_UNUSABLE = frozenset({FailureKind.NOT_FOUND, FailureKind.PERMISSION_DENIED})


def _unusable_interpreters(snapshot: Snapshot) -> bool:
    """A named interpreter that could not even be started: exit code 3, after the report.

    An interpreter that starts and then crashes is a finding about that environment,
    not missing context, so it does not count.
    """
    return any(
        interpreter.version is None and any(f.kind in _UNUSABLE for f in interpreter.failures)
        for interpreter in snapshot.interpreters
    )


def main() -> None:
    """Console entry point.

    Typer reports usage errors itself (exit code 2) and turns Ctrl+C into 130. Anything
    else that escapes is a bug: print one redacted line instead of a traceback, which
    could contain local values, and exit with 2. ``GSI_DEBUG=1`` shows the traceback.
    """
    try:
        app()
    except Exception as exc:  # SystemExit (normal exits) is not an Exception
        if os.environ.get("GSI_DEBUG"):
            raise
        message = Redactor(RedactionContext.current()).text(str(exc)) or type(exc).__name__
        typer.echo(
            f"gpu-stack-inspector: internal error: {message}\n"
            "Run it again with GSI_DEBUG=1 to see the traceback, and please report it.",
            err=True,
        )
        sys.exit(ExitCode.ERROR)
