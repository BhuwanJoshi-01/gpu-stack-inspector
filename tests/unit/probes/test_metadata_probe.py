"""Running the metadata probe and reading its answer (with a scripted runner)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from gpu_stack_inspector.models import Distribution, FailureKind
from gpu_stack_inspector.runner import CommandResult, Outcome
from tests.fakes import FakeRunner

pending = pytest.mark.xfail(strict=True, reason="GPU-003: running the metadata probe pending")

GOOD = {
    "protocol": 1,
    "errors": [],
    "python": {
        "version": "3.12.13",
        "implementation": "cpython",
        "executable": "/envs/a/bin/python",
        "prefix": "/envs/a",
        "base_prefix": "/usr",
        "conda": False,
        "platform": "linux-x86_64",
        "user_site": False,
        "pythonpath_set": False,
    },
    "distributions": [
        {"name": "torch", "version": "2.11.0+cu128", "location": "/envs/a/lib/site-packages"},
        {
            "name": "nvidia-cublas-cu12",
            "version": "12.8.4.1",
            "location": "/envs/a/lib/site-packages",
        },
    ],
}


def module() -> Any:
    from gpu_stack_inspector.probes import metadata_probe

    return metadata_probe


def interpreter(tmp_path: Path) -> Path:
    program = tmp_path / "envs" / "a" / "bin" / "python"
    program.parent.mkdir(parents=True)
    program.write_text("#!/bin/sh\n")
    program.chmod(0o755)
    return program


def wrapped(document: object, noise: str = "") -> str:
    return f"{noise}GSI-PROBE-BEGIN{json.dumps(document)}GSI-PROBE-END\n"


def run(tmp_path: Path, **answer: Any) -> tuple[Any, FakeRunner]:
    python = interpreter(tmp_path)
    argv = (str(python), "-c", module().program_source())
    runner = FakeRunner()
    if "result" in answer:
        runner.script_result(CommandResult(argv, **answer["result"]))
    else:
        runner.script(*argv, **answer)
    return module().probe_interpreter(runner, "A", str(python), cwd=tmp_path), runner


@pending
def test_a_healthy_interpreter_is_described(tmp_path: Path) -> None:
    facts, _ = run(tmp_path, stdout=wrapped(GOOD))
    assert facts.label == "A"
    assert facts.path == str(tmp_path / "envs" / "a" / "bin" / "python")
    assert (facts.version, facts.implementation, facts.executable) == (
        "3.12.13",
        "cpython",
        "/envs/a/bin/python",
    )
    assert facts.virtualenv is True
    assert facts.conda is False
    assert facts.distributions == (
        Distribution("torch", "2.11.0+cu128", "/envs/a/lib/site-packages"),
        Distribution("nvidia-cublas-cu12", "12.8.4.1", "/envs/a/lib/site-packages"),
    )
    assert facts.failures == ()


@pending
def test_the_probe_runs_in_an_empty_temporary_directory_with_a_timeout(tmp_path: Path) -> None:
    _, runner = run(tmp_path, stdout=wrapped(GOOD))
    call = runner.calls[0]
    assert call.cwd is not None
    assert call.cwd != tmp_path
    assert str(call.cwd).startswith(tempfile.gettempdir())
    assert not call.cwd.exists()  # removed again afterwards
    assert call.env_overrides is None  # the target's environment is left as it is
    assert call.timeout == 8.0


@pending
def test_an_unusable_path_runs_nothing(tmp_path: Path) -> None:
    facts = module().probe_interpreter(FakeRunner(), "B", str(tmp_path / "missing"), cwd=tmp_path)
    assert facts.version is None
    assert [f.kind for f in facts.failures] == [FailureKind.NOT_FOUND]


@pending
def test_output_around_the_markers_is_ignored(tmp_path: Path) -> None:
    noise = "sitecustomize: welcome!\nGSI-PROBE-END stray marker\n"
    facts, _ = run(tmp_path, stdout=wrapped(GOOD, noise=noise) + "atexit says bye\n")
    assert facts.version == "3.12.13"
    assert facts.failures == ()


@pending
def test_no_probe_result_is_malformed_output(tmp_path: Path) -> None:
    facts, _ = run(tmp_path, stdout="hello from a strange interpreter\n")
    assert [f.kind for f in facts.failures] == [FailureKind.MALFORMED_OUTPUT]
    assert "no probe result" in facts.failures[0].detail


@pending
def test_a_result_cut_off_by_the_output_limit_says_so(tmp_path: Path) -> None:
    result = {
        "outcome": Outcome.OK,
        "returncode": 0,
        "stdout": "GSI-PROBE-BEGIN{" + "x" * 100,
        "stdout_truncated": True,
    }
    facts, _ = run(tmp_path, result=result)
    assert [f.kind for f in facts.failures] == [FailureKind.MALFORMED_OUTPUT]
    assert "output limit" in facts.failures[0].detail


@pending
def test_invalid_json_is_malformed_output(tmp_path: Path) -> None:
    facts, _ = run(tmp_path, stdout="GSI-PROBE-BEGIN{not json GSI-PROBE-END")
    assert [f.kind for f in facts.failures] == [FailureKind.MALFORMED_OUTPUT]


@pending
def test_an_unknown_protocol_is_malformed_output(tmp_path: Path) -> None:
    facts, _ = run(tmp_path, stdout=wrapped({**GOOD, "protocol": 2}))
    assert facts.version is None
    assert "protocol 2" in facts.failures[0].detail


@pending
def test_fields_of_the_wrong_type_become_unknown(tmp_path: Path) -> None:
    document = json.loads(json.dumps(GOOD))
    document["python"]["version"] = 3.12
    document["python"]["conda"] = "yes"
    document["distributions"].append({"name": "jax", "version": None})
    document["distributions"].append("not an object")
    facts, _ = run(tmp_path, stdout=wrapped(document))
    assert facts.version is None
    assert facts.conda is None
    assert facts.prefix == "/envs/a"
    assert [d.name for d in facts.distributions] == ["torch", "nvidia-cublas-cu12"]
    assert [f.kind for f in facts.failures] == [FailureKind.MALFORMED_OUTPUT]
    assert "conda" in facts.failures[0].detail
    assert "version" in facts.failures[0].detail


@pending
def test_errors_reported_by_the_probe_are_kept(tmp_path: Path) -> None:
    document = {
        **GOOD,
        "distributions": None,
        "errors": ["distributions: OSError: [Errno 5] I/O error"],
    }
    facts, _ = run(tmp_path, stdout=wrapped(document))
    assert facts.version == "3.12.13"
    assert facts.distributions == ()
    assert [(f.kind, f.detail) for f in facts.failures] == [
        (FailureKind.PROBE_ERROR, "distributions: OSError: [Errno 5] I/O error")
    ]


@pending
def test_a_crash_is_explained_by_its_exception_line(tmp_path: Path) -> None:
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "<string>", line 12, in <module>\n'
        "ModuleNotFoundError: No module named 'site'\n"
    )
    facts, _ = run(tmp_path, stderr=stderr, returncode=1)
    assert [f.kind for f in facts.failures] == [FailureKind.NONZERO_EXIT]
    assert facts.failures[0].detail.endswith(
        "exited with code 1: ModuleNotFoundError: No module named 'site'"
    )


@pending
def test_a_fatal_startup_error_is_explained(tmp_path: Path) -> None:
    stderr = (
        "Fatal Python error: init_fs_encoding: failed to get the Python codec of the "
        "filesystem encoding\nPython runtime state: core initialized\n"
        "ModuleNotFoundError: No module named 'encodings'\n\n"
        "Current thread 0x00007f (most recent call first):\n  <no Python frame>\n"
    )
    facts, _ = run(tmp_path, stderr=stderr, returncode=1)
    assert facts.failures[0].detail.endswith("ModuleNotFoundError: No module named 'encodings'")


@pending
def test_an_interpreter_too_old_for_the_probe_is_named_as_such(tmp_path: Path) -> None:
    stderr = '  File "<string>", line 12\nSyntaxError: future feature annotations is not defined\n'
    facts, _ = run(tmp_path, stderr=stderr, returncode=1)
    assert "needs Python 3.8 or newer" in facts.failures[0].detail


@pending
def test_a_hung_interpreter_is_a_timeout(tmp_path: Path) -> None:
    facts, _ = run(tmp_path, result={"outcome": Outcome.TIMEOUT, "duration": 8.0})
    assert [f.kind for f in facts.failures] == [FailureKind.TIMEOUT]
