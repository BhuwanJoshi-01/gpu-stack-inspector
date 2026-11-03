"""The built-in rules as a whole."""

from __future__ import annotations

from gpu_stack_inspector.models import HostFacts, Status
from gpu_stack_inspector.orchestrator import inspect_host
from gpu_stack_inspector.rules.catalog import builtin_rules
from gpu_stack_inspector.rules.engine import evaluate
from tests.fakes import FakeRunner, read_fixture

SMI = "/usr/bin/nvidia-smi"


def test_every_builtin_rule_has_a_title_and_an_explanation() -> None:
    rules = list(builtin_rules())
    assert rules
    for rule in rules:
        assert rule.title
        assert len(rule.explanation) > 40, rule.id
        for source in rule.sources:
            assert source.startswith("https://"), (rule.id, source)


def test_the_real_rtx_3050_laptop_passes_every_host_check() -> None:
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
    snapshot = inspect_host(
        runner,
        which={"nvidia-smi": SMI}.get,
        environ={},
        toolkit_roots=(),
        host=lambda: HostFacts("Linux", "x86_64", "7.0.0-42-generic", "Ubuntu 26.04.1 LTS"),
    )
    host = [f for f in evaluate(snapshot, builtin_rules()) if f.subject == "host"]
    assert [(f.id, f.status) for f in host if f.id.startswith("GSI-SYS")] == [
        ("GSI-SYS-001", Status.PASS),
        ("GSI-SYS-002", Status.PASS),
        ("GSI-SYS-003", Status.PASS),
    ]
