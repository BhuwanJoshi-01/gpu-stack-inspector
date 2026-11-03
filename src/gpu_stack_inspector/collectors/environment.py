"""Allowlisted CUDA-related environment variables of the tool's own process.

Target interpreters inherit them, so they explain differences such as a GPU hidden by
``CUDA_VISIBLE_DEVICES``. Nothing outside the allowlist is ever read: the rest of the
environment can hold tokens and passwords.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from gpu_stack_inspector.models import EnvironmentVariable

ALLOWLIST: tuple[str, ...] = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "CUDA_HOME",
    "CUDA_PATH",
    "LD_LIBRARY_PATH",
    "PYTORCH_CUDA_ALLOC_CONF",
)


def collect_cuda_environment(
    environ: Mapping[str, str] = os.environ,
) -> tuple[EnvironmentVariable, ...]:
    """The allowlisted variables that are set, in allowlist order."""
    return tuple(EnvironmentVariable(name, environ[name]) for name in ALLOWLIST if name in environ)
