"""The program that runs inside target interpreters."""

from __future__ import annotations

import ast
import json
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

from gpu_stack_inspector.runner import Outcome, SubprocessRunner

SOURCE = files("gpu_stack_inspector.probes").joinpath("metadata_program.py").read_text()
STDLIB_ONLY = {
    "__future__",
    "json",
    "os",
    "site",
    "sys",
    "sysconfig",
    "re",
    "importlib",
    "importlib.metadata",
    "typing",
}


def run_probe(**kwargs: object) -> dict[str, object]:
    result = SubprocessRunner().run([sys.executable, "-c", SOURCE], timeout=30, **kwargs)  # type: ignore[arg-type]
    assert result.outcome is Outcome.OK, result.stderr
    begin, end = result.stdout.index("GSI-PROBE-BEGIN"), result.stdout.index("GSI-PROBE-END")
    document: dict[str, object] = json.loads(result.stdout[begin + len("GSI-PROBE-BEGIN") : end])
    return document


def test_the_program_is_valid_python_3_8() -> None:
    ast.parse(SOURCE, feature_version=(3, 8))


def test_the_program_imports_only_the_standard_library() -> None:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported <= STDLIB_ONLY, imported - STDLIB_ONLY


def test_the_program_reports_the_interpreter_it_runs_in() -> None:
    document = run_probe()
    assert document["protocol"] == 1
    python = document["python"]
    assert isinstance(python, dict)
    assert python["version"] == ".".join(str(part) for part in sys.version_info[:3])
    assert python["executable"] == sys.executable
    assert python["prefix"] == sys.prefix
    assert python["base_prefix"] == sys.base_prefix
    assert python["implementation"] == sys.implementation.name
    assert isinstance(python["conda"], bool)


def test_pythonpath_is_reported_as_set_or_not() -> None:
    assert (
        run_probe(env_overrides={"PYTHONPATH": "/nonexistent"})["python"]["pythonpath_set"] is True
    )  # type: ignore[index]


def fake_site_packages(root: Path, *dists: tuple[str, str, str]) -> Path:
    """dist-info folders (folder name, Name, Version) like pip leaves them."""
    site = root / "site-packages"
    for folder, name, version in dists:
        info = site / folder
        info.mkdir(parents=True)
        info.joinpath("METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8"
        )
    return site


def test_allowlisted_distributions_are_listed_with_their_location(tmp_path: Path) -> None:
    site = fake_site_packages(
        tmp_path,
        ("torch-2.11.0+cu128.dist-info", "torch", "2.11.0+cu128"),
        ("nvidia_cublas_cu12-12.8.4.1.dist-info", "nvidia-cublas-cu12", "12.8.4.1"),
        ("Nvidia_Cudnn_CU12-9.10.2.21.dist-info", "Nvidia_Cudnn_CU12", "9.10.2.21"),
        ("requests-2.32.3.dist-info", "requests", "2.32.3"),
        ("my_private_project-0.1.dist-info", "my-private-project", "0.1"),
    )
    listed = run_probe(env_overrides={"PYTHONPATH": str(site)})["distributions"]
    assert isinstance(listed, list)
    from_site = [(d["name"], d["version"]) for d in listed if Path(d["location"]) == site]
    assert from_site == [
        ("nvidia-cublas-cu12", "12.8.4.1"),
        ("nvidia-cudnn-cu12", "9.10.2.21"),
        ("torch", "2.11.0+cu128"),
    ]


def test_ordinary_packages_of_the_test_environment_are_not_listed() -> None:
    names = {d["name"] for d in run_probe()["distributions"]}  # type: ignore[attr-defined]
    assert not names & {"pytest", "typer", "rich", "hypothesis", "gpu-stack-inspector"}


def program() -> dict[str, Any]:
    """The program's functions, loaded without running main()."""
    namespace: dict[str, Any] = {"__name__": "gsi_probe_under_test"}
    exec(compile(SOURCE, "metadata_program.py", "exec"), namespace)  # noqa: S102 - our own file
    return namespace


def test_the_dist_info_fallback_reads_allowlisted_metadata(tmp_path: Path) -> None:
    site = fake_site_packages(
        tmp_path,
        ("torch-1.12.0.dist-info", "torch", "1.12.0"),
        ("six-1.16.0.dist-info", "six", "1.16.0"),
    )
    (site / "broken.dist-info").mkdir()  # no METADATA at all
    found = program()["scan_dist_info"]([str(site), str(tmp_path / "missing")])
    assert found == [{"name": "torch", "version": "1.12.0", "location": str(site)}]


def test_metadata_headers_stop_at_the_first_blank_line(tmp_path: Path) -> None:
    metadata = tmp_path / "METADATA"
    metadata.write_text("Name: torch\nVersion: 2.0\n\nName: not-a-header\n", encoding="utf-8")
    assert program()["read_headers"](str(metadata)) == {"name": "torch", "version": "2.0"}


def test_a_failing_section_is_reported_and_the_others_still_run() -> None:
    def broken() -> object:
        raise OSError("disk on fire")

    result = program()["collect"](
        [("python", lambda: {"version": "3.8"}), ("distributions", broken)]
    )
    assert result["python"] == {"version": "3.8"}
    assert result["distributions"] is None
    assert result["errors"] == ["distributions: OSError: disk on fire"]


def test_a_healthy_interpreter_reports_no_errors() -> None:
    assert run_probe()["errors"] == []


TORCH_METADATA = """Metadata-Version: 2.1
Name: torch
Version: 2.11.0+cu128
Requires-Dist: filelock
Requires-Dist: nvidia-cuda-runtime-cu12==12.8.90; platform_system == "Linux"
Requires-Dist: nvidia_cudnn_cu12==9.10.2.21; platform_system == "Linux"
Requires-Dist: nvidia-cuda-runtime-cu12==12.8.90; platform_system == "Linux"
Requires-Dist: numpy; extra == "numpy"
"""


def fake_torch(root: Path, metadata: str = TORCH_METADATA) -> Path:
    """A torch distribution as pip leaves it, without any real torch code."""
    site = root / "site-packages"
    info = site / "torch-2.11.0+cu128.dist-info"
    info.mkdir(parents=True)
    info.joinpath("METADATA").write_text(metadata, encoding="utf-8")
    (site / "torch" / "lib").mkdir(parents=True)
    return site


def test_without_torch_the_torch_section_is_null() -> None:
    assert run_probe()["torch"] is None


def test_torch_version_location_and_nvidia_requirements_are_reported(tmp_path: Path) -> None:
    site = fake_torch(tmp_path)
    torch = run_probe(env_overrides={"PYTHONPATH": str(site)})["torch"]
    assert isinstance(torch, dict)
    assert torch["version"] == "2.11.0+cu128"
    assert Path(torch["location"]) == site
    assert torch["requires_nvidia"] == ["nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12"]
