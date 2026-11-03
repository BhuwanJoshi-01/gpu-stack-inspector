"""Runs the collectors and assembles the snapshot; no diagnosis happens here."""

from __future__ import annotations

import os
import shutil
import string
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from gpu_stack_inspector import __version__
from gpu_stack_inspector.collectors.environment import collect_cuda_environment
from gpu_stack_inspector.collectors.host import collect_host
from gpu_stack_inspector.collectors.nvidia import collect_nvidia
from gpu_stack_inspector.collectors.toolkit import DEFAULT_ROOTS, collect_toolkits
from gpu_stack_inspector.models import (
    HostFacts,
    InterpreterFacts,
    ProbeFailure,
    Snapshot,
    utc_timestamp,
)
from gpu_stack_inspector.probes.metadata_probe import probe_interpreter
from gpu_stack_inspector.rules.catalog import builtin_rules
from gpu_stack_inspector.rules.engine import RuleContext, evaluate
from gpu_stack_inspector.runner import CommandRunner


def inspect_host(
    runner: CommandRunner,
    *,
    which: Callable[[str], str | None] = shutil.which,
    environ: Mapping[str, str] = os.environ,
    toolkit_roots: Sequence[Path] = DEFAULT_ROOTS,
    host: Callable[[], HostFacts] = collect_host,
) -> Snapshot:
    """Collect the facts about this machine (not yet about any interpreter)."""
    nvidia = collect_nvidia(runner, which=which)
    toolkit = collect_toolkits(runner, which=which, roots=toolkit_roots, environ=environ)
    return Snapshot(
        tool_version=__version__,
        created_at=utc_timestamp(),
        host=host(),
        nvidia=nvidia,
        toolkit=toolkit,
        cuda_environment=collect_cuda_environment(environ),
        probe_failures=(*nvidia.failures, *toolkit.failures),
    )


def inspect(
    runner: CommandRunner,
    pythons: Sequence[str],
    *,
    cwd: Path | None = None,
    which: Callable[[str], str | None] = shutil.which,
    environ: Mapping[str, str] = os.environ,
    toolkit_roots: Sequence[Path] = DEFAULT_ROOTS,
    host: Callable[[], HostFacts] = collect_host,
) -> Snapshot:
    """Inspect the machine and each named interpreter, labelled A, B, C… in order.

    Interpreter failures are added to the snapshot's list tagged with their label, so a
    report can say which environment could not be checked.
    """
    snapshot = inspect_host(
        runner, which=which, environ=environ, toolkit_roots=toolkit_roots, host=host
    )
    interpreters = tuple(
        probe_interpreter(runner, interpreter_label(index), raw, cwd=cwd)
        for index, raw in enumerate(pythons)
    )
    return replace(
        snapshot,
        interpreters=interpreters,
        probe_failures=(*snapshot.probe_failures, *_tagged_failures(interpreters)),
    )


def interpreter_label(index: int) -> str:
    """A, B, … Z, then #27, #28 …"""
    return string.ascii_uppercase[index] if index < 26 else f"#{index + 1}"


def _tagged_failures(interpreters: Sequence[InterpreterFacts]) -> list[ProbeFailure]:
    return [
        replace(failure, probe=f"{failure.probe} {facts.label}")
        for facts in interpreters
        for failure in facts.failures
    ]


def default_interpreters(which: Callable[[str], str | None] = shutil.which) -> list[str]:
    """The interpreter the user's shell would run: ``python``, else ``python3``, on PATH.

    Not the tool's own interpreter: installed with pipx or ``uv tool`` it lives in an
    isolated environment that tells the user nothing about their projects.
    """
    for name in ("python", "python3"):
        found = which(name)
        if found:
            return [found]
    return []


def diagnose(snapshot: Snapshot, context: RuleContext | None = None) -> Snapshot:
    """The snapshot with the findings of every built-in rule attached."""
    return replace(snapshot, findings=evaluate(snapshot, builtin_rules(), context))
