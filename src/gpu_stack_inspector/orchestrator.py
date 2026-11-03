"""Runs the collectors and assembles the snapshot; no diagnosis happens here."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from gpu_stack_inspector import __version__
from gpu_stack_inspector.collectors.environment import collect_cuda_environment
from gpu_stack_inspector.collectors.host import collect_host
from gpu_stack_inspector.collectors.nvidia import collect_nvidia
from gpu_stack_inspector.collectors.toolkit import DEFAULT_ROOTS, collect_toolkits
from gpu_stack_inspector.models import HostFacts, Snapshot, utc_timestamp
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
