"""Render a real `compare` run as an SVG for the README (maintainer tool, not shipped).

    uv run python scripts/render_demo.py --python A --python B [--rename OLD=NEW] OUT.svg

It runs the actual inspection on this machine, redacts it like every report, and lets
Rich export what it printed. --rename shortens paths for the picture (applied like
redaction, to text only); nothing else is changed.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from rich.console import Console

from gpu_stack_inspector import orchestrator
from gpu_stack_inspector.privacy import redact_snapshot
from gpu_stack_inspector.reports.terminal import render_compare
from gpu_stack_inspector.rules.engine import RuleContext
from gpu_stack_inspector.runner import SubprocessRunner


def rename(value: Any, pairs: list[tuple[str, str]]) -> Any:
    if isinstance(value, Enum):
        return value
    if isinstance(value, str):
        for old, new in pairs:
            value = value.replace(old, new)
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{f.name: rename(getattr(value, f.name), pairs) for f in fields(value) if f.init},
        )
    if isinstance(value, tuple):
        return tuple(rename(item, pairs) for item in value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", action="append", required=True)
    parser.add_argument("--rename", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--width", type=int, default=110)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pairs = [tuple(item.split("=", 1)) for item in args.rename]
    snapshot = orchestrator.with_comparison(
        orchestrator.diagnose(orchestrator.inspect(SubprocessRunner(), args.python), RuleContext())
    )
    snapshot = rename(redact_snapshot(snapshot), pairs)
    console = Console(record=True, width=args.width, force_terminal=True)
    render_compare(snapshot, console)
    console.save_svg(str(args.output), title="gpu-stack-inspector compare")


if __name__ == "__main__":
    main()
