import json
import os
from pathlib import Path

import pytest

from gpu_stack_inspector.collectors.toolkit import (
    collect_toolkits,
    find_toolkits,
    parse_nvcc_version,
)
from gpu_stack_inspector.models import CudaToolkit, FailureKind, ToolkitFacts
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
    return collect_toolkits(runner, which=lambda name: nvcc if name == "nvcc" else None, roots=())


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


def make_toolkit(root: Path, name: str, version_json: object | None = None) -> Path:
    directory = root / name
    (directory / "bin").mkdir(parents=True)
    if version_json is not None:
        (directory / "version.json").write_text(json.dumps(version_json))
    return directory


def test_toolkits_under_the_roots_are_found_with_their_version(tmp_path: Path) -> None:
    usr_local, opt = tmp_path / "usr-local", tmp_path / "opt"
    a = make_toolkit(usr_local, "cuda-12.4", {"cuda": {"name": "CUDA SDK", "version": "12.4.1"}})
    b = make_toolkit(opt, "cuda", {"cuda": {"version": "13.0.2"}})
    assert find_toolkits([usr_local, opt])[0] == [
        CudaToolkit(str(a.resolve()), "12.4.1", "version.json"),
        CudaToolkit(str(b.resolve()), "13.0.2", "version.json"),
    ]


def test_unrelated_directories_and_files_are_ignored(tmp_path: Path) -> None:
    for name in ("cudnn", "cuda-samples", "lib", "cuda_backup"):
        (tmp_path / name).mkdir()
    (tmp_path / "cuda-11.8").write_text("a file, not a toolkit")
    assert find_toolkits([tmp_path]) == ([], [])


def test_missing_roots_are_not_an_error(tmp_path: Path) -> None:
    assert find_toolkits([tmp_path / "does-not-exist"]) == ([], [])


@pytest.mark.parametrize("content", ["{not json", "[]", '{"cuda": {"version": 12}}', ""])
def test_unusable_version_json_leaves_the_version_unknown(tmp_path: Path, content: str) -> None:
    directory = make_toolkit(tmp_path, "cuda-12.4")
    (directory / "version.json").write_text(content)
    assert find_toolkits([tmp_path])[0] == [CudaToolkit(str(directory.resolve()))]


def test_huge_version_files_are_read_only_partially(tmp_path: Path) -> None:
    directory = make_toolkit(tmp_path, "cuda-12.4")
    (directory / "version.json").write_text(
        '{"cuda": {"version": "12.4.1"}, "pad": "' + "x" * 200_000 + '"}'
    )
    assert find_toolkits([tmp_path])[0] == [CudaToolkit(str(directory.resolve()))]


def test_old_toolkits_are_versioned_by_version_txt(tmp_path: Path) -> None:
    directory = make_toolkit(tmp_path, "cuda-10.2")
    (directory / "version.txt").write_text("CUDA Version 10.2.89\n")
    assert find_toolkits([tmp_path])[0] == [
        CudaToolkit(str(directory.resolve()), "10.2.89", "version.txt")
    ]


@pytest.mark.parametrize(
    ("define", "version"),
    [("#define CUDA_VERSION 12040", "12.4"), ("#  define CUDA_VERSION 11080 /* 11.8 */", "11.8")],
)
def test_cuda_header_is_the_last_resort(tmp_path: Path, define: str, version: str) -> None:
    directory = make_toolkit(tmp_path, "cuda")
    (directory / "include").mkdir()
    (directory / "include" / "cuda.h").write_text(f"/* header */\n{define}\n")
    assert find_toolkits([tmp_path])[0] == [
        CudaToolkit(str(directory.resolve()), version, "include/cuda.h")
    ]


def test_version_json_wins_over_the_other_sources(tmp_path: Path) -> None:
    directory = make_toolkit(tmp_path, "cuda-12.4", {"cuda": {"version": "12.4.1"}})
    (directory / "version.txt").write_text("CUDA Version 9.0.0\n")
    assert find_toolkits([tmp_path])[0][0].version == "12.4.1"


posix_only = pytest.mark.skipif(
    os.name != "posix", reason="creating symlinks needs privileges on Windows"
)


@posix_only
def test_a_symlink_to_a_toolkit_is_an_alias_not_a_second_toolkit(tmp_path: Path) -> None:
    real = make_toolkit(tmp_path, "cuda-12.4", {"cuda": {"version": "12.4.1"}})
    (tmp_path / "cuda").symlink_to(real)
    toolkits, failures = find_toolkits([tmp_path])
    assert toolkits == [
        CudaToolkit(str(real.resolve()), "12.4.1", "version.json", (str(tmp_path / "cuda"),))
    ]
    assert failures == []


@posix_only
def test_a_dangling_symlink_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "cuda").symlink_to(tmp_path / "cuda-11.8")
    assert find_toolkits([tmp_path]) == ([], [])


def test_cuda_home_outside_the_roots_is_included(tmp_path: Path) -> None:
    elsewhere = make_toolkit(tmp_path / "home", "my-cuda", {"cuda": {"version": "12.8.0"}})
    toolkits, _ = find_toolkits([], {"CUDA_HOME": str(elsewhere)})
    assert [t.path for t in toolkits] == [str(elsewhere.resolve())]


def test_a_toolkit_found_twice_is_listed_once(tmp_path: Path) -> None:
    directory = make_toolkit(tmp_path, "cuda-12.4", {"cuda": {"version": "12.4.1"}})
    env = {
        "CUDA_HOME": str(directory),
        "CUDA_PATH": str(directory),
        "CUDA_PATH_V12_4": str(directory),
    }
    toolkits, failures = find_toolkits([tmp_path], env)
    assert len(toolkits) == 1
    assert failures == []


def test_a_variable_naming_a_missing_directory_is_reported(tmp_path: Path) -> None:
    missing = tmp_path / "cuda-11.8"
    toolkits, failures = find_toolkits([], {"CUDA_HOME": str(missing), "CUDA_PATH": ""})
    assert toolkits == []
    assert [(f.probe, f.kind) for f in failures] == [("cuda-toolkits", FailureKind.NOT_FOUND)]
    assert failures[0].detail == f"CUDA_HOME names {missing}, which is not a directory"


def test_collect_toolkits_combines_directories_and_nvcc(tmp_path: Path) -> None:
    directory = make_toolkit(tmp_path, "cuda-12.4", {"cuda": {"version": "12.4.1"}})
    runner = FakeRunner()
    runner.script(NVCC, "--version", stdout=read_fixture("nvcc", "ubuntu-12.4.version.txt"))
    facts = collect_toolkits(runner, which=lambda _: NVCC, roots=[tmp_path], environ={})
    assert facts.nvcc_version == "12.4.131"
    assert [t.path for t in facts.toolkits] == [str(directory.resolve())]
