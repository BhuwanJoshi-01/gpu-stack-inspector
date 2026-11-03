import pytest

from gpu_stack_inspector.collectors.toolkit import parse_nvcc_version
from tests.fakes import read_fixture


@pytest.mark.parametrize(
    ("fixture", "version"),
    [("ubuntu-12.4.version.txt", "12.4.131"), ("old-10.2.version.txt", "10.2.89")],
)
def test_nvcc_version_is_read_from_the_release_line(fixture: str, version: str) -> None:
    assert parse_nvcc_version(read_fixture("nvcc", fixture)) == version


@pytest.mark.parametrize("text", ["", "nvcc: command not found", "release twelve"])
def test_unrecognised_nvcc_output_has_no_version(text: str) -> None:
    assert parse_nvcc_version(text) is None
