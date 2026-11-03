"""Real probe answers from conda environments on one laptop (see the fixtures README)."""

from __future__ import annotations

import pytest

from gpu_stack_inspector.models import InterpreterFacts
from gpu_stack_inspector.probes.metadata_probe import parse_probe_output
from tests.fakes import read_fixture


def parse(env: str) -> InterpreterFacts:
    stdout = read_fixture("probe", f"conda-{env}.stdout.txt")
    return parse_probe_output("A", f"~/miniconda3/envs/{env}/bin/python", stdout)


def versions(facts: InterpreterFacts) -> dict[str, str]:
    return {d.name: d.version for d in facts.distributions}


def nvidia_wheels(facts: InterpreterFacts) -> int:
    return sum(d.name.startswith("nvidia-") for d in facts.distributions)


@pytest.mark.parametrize("env", ["base", "sapiens2", "wilor", "smplerx", "dwpose"])
def test_every_real_answer_parses_cleanly_as_a_conda_env(env: str) -> None:
    facts = parse(env)
    assert facts.failures == ()
    assert facts.conda is True
    assert facts.virtualenv is False
    assert facts.implementation == "cpython"
    assert facts.executable is not None
    assert facts.executable.startswith("~/miniconda3/")


def test_a_wheel_from_the_pytorch_index_has_a_cuda_tag_and_nvidia_wheels() -> None:
    facts = parse("sapiens2")
    assert facts.version == "3.13.13"
    assert versions(facts)["torch"] == "2.11.0+cu128"
    assert nvidia_wheels(facts) == 15


def test_a_cuda_wheel_from_pypi_has_no_tag_but_nvidia_wheels() -> None:
    facts = parse("wilor")
    assert versions(facts)["torch"] == "2.14.1"
    assert nvidia_wheels(facts) == 15


def test_a_conda_installed_torch_on_python_3_8_has_neither() -> None:
    facts = parse("smplerx")
    assert facts.version == "3.8.20"
    assert versions(facts)["torch"] == "1.12.0"
    assert nvidia_wheels(facts) == 0


def test_packages_that_need_torch_can_be_installed_without_it() -> None:
    names = versions(parse("base"))
    assert "torch" not in names
    assert {"accelerate", "bitsandbytes"} <= names.keys()


def test_gpu_wheels_can_come_from_onnxruntime_instead_of_torch() -> None:
    facts = parse("dwpose")
    assert "torch" not in versions(facts)
    assert "onnxruntime-gpu" in versions(facts)
    assert nvidia_wheels(facts) == 7


def test_torch_evidence_of_a_pytorch_index_cuda_wheel() -> None:
    torch = parse("sapiens2").torch
    assert torch is not None
    assert (torch.version, torch.built_cuda, torch.cuda_library) == ("2.11.0+cu128", "12.8", True)
    assert "nvidia-cudnn-cu12" in torch.requires_nvidia
    assert torch.conda_records == ()  # a conda env, but torch came from pip


def test_torch_evidence_of_an_untagged_pypi_cuda_wheel() -> None:
    torch = parse("wilor").torch
    assert torch is not None
    assert (torch.version, torch.built_cuda, torch.cuda_library) == ("2.14.1", "13.0", True)
    assert "nvidia-cudnn-cu13" in torch.requires_nvidia


def test_torch_evidence_of_a_conda_cuda_build() -> None:
    torch = parse("smplerx").torch
    assert torch is not None
    assert (torch.version, torch.built_cuda, torch.cuda_library) == ("1.12.0", "11.3", True)
    assert torch.requires_nvidia == ()
    assert torch.conda_records is not None
    assert "pytorch-1.12.0-py3.8_cuda11.3_cudnn8.3.2_0" in torch.conda_records
    assert "pytorch-mutex-1.0-cuda" in torch.conda_records


@pytest.mark.parametrize("env", ["base", "dwpose"])
def test_environments_without_torch_have_no_torch_evidence(env: str) -> None:
    assert parse(env).torch is None
