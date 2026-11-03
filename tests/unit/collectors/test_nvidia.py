"""nvidia-smi collector, driven by real and synthetic captures."""

from __future__ import annotations

from typing import Any

import pytest

from gpu_stack_inspector.models import FailureKind, GpuDevice
from gpu_stack_inspector.runner import CommandResult, Outcome
from tests.fakes import FakeRunner, UnexpectedCommandError, read_fixture

SMI = "/usr/bin/nvidia-smi"
FIELDS = "index,name,driver_version,memory.total,compute_cap"
QUERY = (SMI, f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits")
QUERY_WITHOUT_CC = (
    SMI,
    "--query-gpu=index,name,driver_version,memory.total",
    "--format=csv,noheader,nounits",
)
VERSION = (SMI, "--version")
TABLE = (SMI,)


def collect(runner: FakeRunner, smi: str | None = SMI) -> Any:
    from gpu_stack_inspector.collectors.nvidia import collect_nvidia

    return collect_nvidia(runner, which=lambda name: smi if name == "nvidia-smi" else None)


def fixture(name: str) -> str:
    return read_fixture("nvidia-smi", name)


def scripted(*, query: str | None = None, version: str | None = None) -> FakeRunner:
    runner = FakeRunner()
    runner.script(
        *QUERY, stdout=query if query is not None else fixture("rtx3050-laptop-595.query.csv")
    )
    runner.script(
        *VERSION,
        stdout=version if version is not None else fixture("rtx3050-laptop-595.version.txt"),
    )
    return runner


def test_without_nvidia_smi_nothing_runs_and_nothing_failed() -> None:
    facts = collect(FakeRunner(), smi=None)  # any command would raise UnexpectedCommandError
    assert facts.smi_path is None
    assert facts.gpus == ()
    assert facts.failures == ()


def test_real_rtx_3050_laptop() -> None:
    runner = scripted()
    facts = collect(runner)
    assert facts.smi_path == SMI
    assert facts.gpus == (
        GpuDevice(
            0, "NVIDIA GeForce RTX 3050 Laptop GPU", memory_total_mib=4096, compute_capability="8.6"
        ),
    )
    assert facts.driver_version == "595.99.02"
    assert facts.max_cuda_version == "13.2"
    assert facts.failures == ()


def test_commands_run_with_the_c_locale_and_a_timeout() -> None:
    runner = scripted()
    collect(runner)
    assert [call.argv for call in runner.calls] == [QUERY, VERSION]
    for call in runner.calls:
        assert call.env_overrides == {"LC_ALL": "C"}
        assert 0 < call.timeout <= 10


def test_two_different_gpus() -> None:
    facts = collect(scripted(query=fixture("two-gpus.query.csv")))
    assert [(g.index, g.name, g.memory_total_mib, g.compute_capability) for g in facts.gpus] == [
        (0, "NVIDIA GeForce RTX 4090", 24564, "8.9"),
        (1, "NVIDIA A100-SXM4-80GB", 81920, "8.0"),
    ]


def test_values_reported_as_not_available_are_unknown() -> None:
    facts = collect(scripted(query=fixture("not-available.query.csv")))
    assert facts.gpus == (GpuDevice(0, "NVIDIA GeForce GTX 1050", None, None),)
    assert facts.driver_version == "470.256.02"


def test_driver_not_loaded_is_a_failure_and_stops_further_queries() -> None:
    runner = FakeRunner()
    runner.script(*QUERY, stdout=fixture("driver-not-loaded.stdout.txt"), returncode=9)
    facts = collect(runner)
    assert facts.gpus == ()
    assert len(facts.failures) == 1
    failure = facts.failures[0]
    assert failure.probe == "nvidia-smi"
    assert failure.kind is FailureKind.NONZERO_EXIT
    assert "couldn't communicate with the NVIDIA driver" in failure.detail
    assert [call.argv for call in runner.calls] == [QUERY]


def test_no_devices_found_means_zero_gpus_not_a_failure() -> None:
    runner = FakeRunner()
    runner.script(*QUERY, stdout=fixture("no-devices.stdout.txt"), returncode=6)
    runner.script(*VERSION, stdout=fixture("rtx3050-laptop-595.version.txt"))
    facts = collect(runner)
    assert facts.gpus == ()
    assert facts.failures == ()
    assert facts.driver_version == "595.99.02"
    assert facts.max_cuda_version == "13.2"


def test_a_hung_nvidia_smi_is_a_timeout_failure() -> None:
    runner = FakeRunner()
    runner.script_result(CommandResult(QUERY, Outcome.TIMEOUT, duration=8.0))
    facts = collect(runner)
    assert [f.kind for f in facts.failures] == [FailureKind.TIMEOUT]


def test_old_driver_without_compute_cap_is_queried_again_without_it() -> None:
    runner = FakeRunner()
    error = fixture("invalid-field.stdout.txt").replace("bogus_field", "compute_cap")
    runner.script(*QUERY, stdout=error, returncode=2)
    runner.script(*QUERY_WITHOUT_CC, stdout="0, Tesla K80, 470.256.02, 11441\n")
    runner.script(*VERSION, stdout=fixture("rtx3050-laptop-595.version.txt"))
    facts = collect(runner)
    assert facts.gpus == (GpuDevice(0, "Tesla K80", 11441, None),)
    assert facts.failures == ()


def test_driver_without_version_flag_falls_back_to_the_table_header() -> None:
    runner = scripted()
    runner.script(*VERSION, stdout="Invalid combination of input arguments.\n", returncode=2)
    runner.script(*TABLE, stdout=fixture("rtx3050-laptop-595.header.txt"))
    facts = collect(runner)
    assert facts.max_cuda_version == "13.2"
    assert facts.driver_version == "595.99.02"
    assert facts.failures == ()


def test_unknown_max_cuda_version_is_left_unknown_when_both_sources_fail() -> None:
    runner = scripted()
    runner.script(*VERSION, stdout="Invalid combination of input arguments.\n", returncode=2)
    runner.script(*TABLE, stdout="something unexpected\n")
    facts = collect(runner)
    assert facts.max_cuda_version is None
    assert facts.gpus  # the GPU list is still reported
    assert [f.detail for f in facts.failures] == [
        "nvidia-smi did not report the driver's maximum CUDA version"
    ]


def test_malformed_rows_are_skipped_and_reported() -> None:
    query = fixture("two-gpus.query.csv") + "garbage without commas\n2, broken, row\n"
    facts = collect(scripted(query=query))
    assert [g.index for g in facts.gpus] == [0, 1]
    assert [f.kind for f in facts.failures] == [
        FailureKind.MALFORMED_OUTPUT,
        FailureKind.MALFORMED_OUTPUT,
    ]


def test_only_the_scripted_read_only_commands_are_run() -> None:
    runner = FakeRunner()
    with pytest.raises(UnexpectedCommandError):
        collect(runner)


def test_blank_lines_are_ignored() -> None:
    query = "\n" + fixture("rtx3050-laptop-595.query.csv") + "\n  \n"
    facts = collect(scripted(query=query))
    assert len(facts.gpus) == 1
    assert facts.failures == ()


@pytest.mark.parametrize(
    "row", ["x, NVIDIA GeForce RTX 3050, 595.99.02, 4096, 8.6", "0, [N/A], 595.99.02, 4096, 8.6"]
)
def test_rows_without_a_usable_index_or_name_are_malformed(row: str) -> None:
    facts = collect(scripted(query=row + "\n"))
    assert facts.gpus == ()
    assert [f.detail for f in facts.failures] == ["line 1 has no GPU index or name"]


def test_output_the_csv_module_cannot_read_is_malformed() -> None:
    huge_field = "x" * 200_000  # beyond csv's 128 KiB field limit, within the output cap
    facts = collect(scripted(query=f"0, {huge_field}, 595.99.02, 4096, 8.6\n"))
    assert facts.gpus == ()
    assert facts.failures[0].kind is FailureKind.MALFORMED_OUTPUT
    assert facts.failures[0].detail.startswith("unreadable CSV output")
