import pytest

from gpu_stack_inspector.models import FailureKind, ProbeFailure
from gpu_stack_inspector.runner import CommandResult, Outcome, probe_failure


def test_only_ok_outcome_counts_as_ok() -> None:
    assert CommandResult(("true",), Outcome.OK, returncode=0).ok
    for outcome in Outcome:
        if outcome is not Outcome.OK:
            assert not CommandResult(("x",), outcome).ok


def test_successful_result_is_not_a_failure() -> None:
    assert probe_failure("nvidia-smi", CommandResult(("nvidia-smi",), Outcome.OK, 0)) is None


@pytest.mark.parametrize(
    ("result", "kind", "detail"),
    [
        (
            CommandResult(("nvidia-smi",), Outcome.NOT_FOUND, error="No such file"),
            FailureKind.NOT_FOUND,
            "nvidia-smi was not found",
        ),
        (
            CommandResult(("nvcc",), Outcome.PERMISSION),
            FailureKind.PERMISSION_DENIED,
            "permission denied when starting nvcc",
        ),
        (
            CommandResult(("nvidia-smi",), Outcome.TIMEOUT, duration=8.04),
            FailureKind.TIMEOUT,
            "nvidia-smi did not finish within 8.0 s and was stopped",
        ),
        (
            CommandResult(
                ("nvidia-smi",),
                Outcome.NONZERO,
                returncode=9,
                stderr="\nNVIDIA-SMI has failed because it couldn't communicate\nsecond line\n",
            ),
            FailureKind.NONZERO_EXIT,
            "nvidia-smi exited with code 9: NVIDIA-SMI has failed because it couldn't communicate",
        ),
        (
            CommandResult(("python",), Outcome.OS_ERROR, error="Exec format error"),
            FailureKind.OS_ERROR,
            "python could not be started: Exec format error",
        ),
    ],
)
def test_failed_results_become_probe_failures(
    result: CommandResult, kind: FailureKind, detail: str
) -> None:
    failure = probe_failure("gpu", result)
    assert failure == ProbeFailure("gpu", kind, detail)


def test_failure_detail_is_kept_short() -> None:
    result = CommandResult(("x",), Outcome.NONZERO, returncode=1, stderr="e" * 5000)
    failure = probe_failure("x", result)
    assert failure is not None
    assert len(failure.detail) == 200
    assert failure.detail.endswith("…")


def test_failure_detail_uses_stdout_when_stderr_is_empty() -> None:
    result = CommandResult(
        ("nvidia-smi",),
        Outcome.NONZERO,
        returncode=9,
        stdout="NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n",
    )
    failure = probe_failure("nvidia-smi", result)
    assert failure is not None
    assert failure.detail == (
        "nvidia-smi exited with code 9: "
        "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."
    )


def test_failure_detail_prefers_stderr() -> None:
    result = CommandResult(("x",), Outcome.NONZERO, returncode=1, stdout="out", stderr="err")
    failure = probe_failure("x", result)
    assert failure is not None
    assert failure.detail == "x exited with code 1: err"
