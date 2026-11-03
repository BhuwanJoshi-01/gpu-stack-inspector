"""Command-line interface: parses arguments and wires the layers together.

No diagnosis logic lives here; commands call collectors, rules and renderers.
"""

from __future__ import annotations

import os
import sys
from enum import StrEnum
from typing import Annotated

import typer
from rich.console import Console

from gpu_stack_inspector import __version__, orchestrator
from gpu_stack_inspector.exit_codes import ExitCode
from gpu_stack_inspector.privacy import RedactionContext, Redactor
from gpu_stack_inspector.reports.json_report import render_json
from gpu_stack_inspector.reports.terminal import render_terminal
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
    pythons = python or orchestrator.default_interpreters()
    snapshot = orchestrator.inspect(SubprocessRunner(), pythons)
    redacted = Redactor(RedactionContext.current()).redact(snapshot)
    if output_format is OutputFormat.JSON:
        typer.echo(render_json(redacted), nl=False)
    else:
        render_terminal(redacted, Console())


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
