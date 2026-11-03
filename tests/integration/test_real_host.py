"""Runs the collectors against the real machine. Opt in with GSI_INTEGRATION=1.

These tests only read: they run nvidia-smi and nvcc queries like any inspection.
"""

from __future__ import annotations

import os

import pytest

from gpu_stack_inspector.orchestrator import inspect_host
from gpu_stack_inspector.runner import SubprocessRunner

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("GSI_INTEGRATION"), reason="set GSI_INTEGRATION=1"),
]


def test_real_host_inspection_completes_without_errors() -> None:
    snapshot = inspect_host(SubprocessRunner())
    assert snapshot.host is not None
    assert snapshot.nvidia is not None
    if snapshot.nvidia.smi_path is not None and not snapshot.nvidia.failures:
        assert snapshot.nvidia.driver_version
        for gpu in snapshot.nvidia.gpus:
            assert gpu.name
