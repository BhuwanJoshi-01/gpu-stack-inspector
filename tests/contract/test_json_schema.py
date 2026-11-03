"""The JSON output is a contract: it must match docs/json-schema-v1.json exactly."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from gpu_stack_inspector import models
from gpu_stack_inspector.models import HostFacts, Snapshot
from gpu_stack_inspector.orchestrator import diagnose, inspect_host, with_comparison
from gpu_stack_inspector.probes.metadata_probe import parse_probe_output
from gpu_stack_inspector.reports.json_report import render_json
from gpu_stack_inspector.rules.torch_build import TorchBuild
from tests.fakes import FakeRunner, read_fixture

SCHEMA = json.loads((Path(__file__).parents[2] / "docs" / "json-schema-v1.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)
SMI = "/usr/bin/nvidia-smi"


def real_laptop_snapshot() -> Snapshot:
    """The RTX 3050 laptop: real nvidia-smi output and real probe answers, every check run."""
    runner = FakeRunner()
    runner.script(
        SMI,
        "--query-gpu=index,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
        stdout=read_fixture("nvidia-smi", "rtx3050-laptop-595.query.csv"),
    )
    runner.script(
        SMI, "--version", stdout=read_fixture("nvidia-smi", "rtx3050-laptop-595.version.txt")
    )
    host = inspect_host(
        runner,
        which={"nvidia-smi": SMI}.get,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        toolkit_roots=(),
        host=lambda: HostFacts("Linux", "x86_64", "7.0.0-42-generic", "Ubuntu 26.04.1 LTS"),
    )
    interpreters = tuple(
        parse_probe_output(label, f"~/{name}/bin/python", read_fixture("probe", fixture))
        for label, name, fixture in (
            ("A", "sapiens2", "conda-sapiens2.stdout.txt"),
            ("B", "cpu-venv", "venv-cpu-torch.stdout.txt"),
            ("C", "base", "conda-base.stdout.txt"),
        )
    )
    return with_comparison(diagnose(dataclasses.replace(host, interpreters=interpreters)))


def validate(document: Any) -> list[str]:
    return [
        f"{'/'.join(map(str, e.absolute_path))}: {e.message}"
        for e in VALIDATOR.iter_errors(document)
    ]


def test_the_schema_itself_is_valid() -> None:
    Draft202012Validator.check_schema(SCHEMA)


def test_real_output_matches_the_schema() -> None:
    document = json.loads(render_json(real_laptop_snapshot()))
    assert validate(document) == []
    assert document["findings"]  # the checks ran
    assert document["comparison"]  # three interpreters were compared


def test_unknown_fields_are_rejected_so_changes_cannot_slip_in() -> None:
    document = json.loads(render_json(real_laptop_snapshot()))
    document["interpreters"][0]["new_field"] = "x"
    assert validate(document)


SCHEMA_DEFS = {
    models.Snapshot: SCHEMA,
    models.HostFacts: SCHEMA["$defs"]["host"],
    models.NvidiaFacts: SCHEMA["$defs"]["nvidia"],
    models.GpuDevice: SCHEMA["$defs"]["gpu"],
    models.ToolkitFacts: SCHEMA["$defs"]["toolkit"],
    models.CudaToolkit: SCHEMA["$defs"]["cuda_toolkit"],
    models.EnvironmentVariable: SCHEMA["$defs"]["environment_variable"],
    models.InterpreterFacts: SCHEMA["$defs"]["interpreter"],
    models.Distribution: SCHEMA["$defs"]["distribution"],
    models.TorchFacts: SCHEMA["$defs"]["torch"],
    models.FieldComparison: SCHEMA["$defs"]["comparison_row"],
    models.ProbeFailure: SCHEMA["$defs"]["probe_failure"],
    models.Finding: SCHEMA["$defs"]["finding"],
    models.Evidence: SCHEMA["$defs"]["evidence"],
    TorchBuild: SCHEMA["$defs"]["torch_build"],
}
DERIVED = {
    models.InterpreterFacts: {"virtualenv", "torch_build"},
    models.FieldComparison: {"differs"},
}


@pytest.mark.parametrize("model", list(SCHEMA_DEFS), ids=lambda m: m.__name__)
def test_every_model_field_is_in_the_schema_and_nothing_else(model: type) -> None:
    fields = {f.name for f in dataclasses.fields(model)} | DERIVED.get(model, set())
    schema = SCHEMA_DEFS[model]
    assert set(schema["properties"]) == fields
    assert set(schema["required"]) == fields
    assert schema["additionalProperties"] is False
