import pytest

from gpu_stack_inspector.collectors.toolkit import collect_toolkits, parse_nvcc_version
from gpu_stack_inspector.models import FailureKind, ToolkitFacts
from gpu_stack_inspector.runner import CommandResult, Outcome
from tests.fakes import FakeRunner, read_fixture


@pytest.mark.parametrize(
    ("fixture", "version"),
    [("ubuntu-12.4.version.txt", "12.4.131"), ("old-10.2.version.txt", "10.2.89")],
)
def test_nvcc_version_is_read_from_the_release_line(fixture: str, version: str) -> None:
    assert parse_nvcc_version(read_fixture("nvcc", fixture)) == version


@pytest.mark.parametrize("text", ["", "nvcc: command not found", "release twelve"])
def test_unrecognised_nvcc_output_has_no_version(text: str) -> None:
    assert parse_nvcc_version(text) is None


NVCC = "/usr/bin/nvcc"


def _collect(runner: FakeRunner, nvcc: str | None = NVCC) -> ToolkitFacts:
    return collect_toolkits(runner, which=lambda name: nvcc if name == "nvcc" else None)


def test_without_nvcc_nothing_runs() -> None:
    assert _collect(FakeRunner(), nvcc=None) == ToolkitFacts()


def test_nvcc_version_is_collected_with_the_c_locale() -> None:
    runner = FakeRunner()
    runner.script(NVCC, "--version", stdout=read_fixture("nvcc", "ubuntu-12.4.version.txt"))
    facts = _collect(runner)
    assert (facts.nvcc_path, facts.nvcc_version, facts.failures) == (NVCC, "12.4.131", ())
    assert runner.calls[0].env_overrides == {"LC_ALL": "C"}


def test_a_broken_nvcc_is_a_failure() -> None:
    runner = FakeRunner()
    runner.script_result(CommandResult((NVCC, "--version"), Outcome.TIMEOUT, duration=8))
    facts = _collect(runner)
    assert facts.nvcc_version is None
    assert [f.kind for f in facts.failures] == [FailureKind.TIMEOUT]


def test_nvcc_output_without_a_release_is_malformed() -> None:
    runner = FakeRunner()
    runner.script(NVCC, "--version", stdout="something else\n")
    facts = _collect(runner)
    assert [f.kind for f in facts.failures] == [FailureKind.MALFORMED_OUTPUT]
