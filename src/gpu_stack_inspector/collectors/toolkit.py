"""CUDA toolkits on disk and the nvcc on PATH."""

from __future__ import annotations

import re

_NVCC_RELEASE = re.compile(
    r"release\s+(?P<release>[0-9]+(?:\.[0-9]+)*),\s*V(?P<full>[0-9]+(?:\.[0-9]+)*)"
)


def parse_nvcc_version(stdout: str) -> str | None:
    """``Cuda compilation tools, release 12.4, V12.4.131`` → ``12.4.131``."""
    match = _NVCC_RELEASE.search(stdout)
    return match["full"] if match else None
