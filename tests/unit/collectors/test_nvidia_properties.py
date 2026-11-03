"""Whatever nvidia-smi prints, the collector returns facts and never raises."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gpu_stack_inspector.collectors.nvidia import collect_nvidia
from gpu_stack_inspector.models import NvidiaFacts
from gpu_stack_inspector.runner import CommandResult, Outcome

SMI = "/usr/bin/nvidia-smi"

answers = st.tuples(
    st.sampled_from(list(Outcome)),
    st.integers(min_value=0, max_value=255),
    st.text(max_size=300),
)


class ArbitraryNvidiaSmi:
    """Answers every nvidia-smi invocation with generated output."""

    def __init__(self, data: st.DataObject) -> None:
        self._data = data

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        env_overrides: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> CommandResult:
        assert argv[0] == SMI, f"unexpected program {argv[0]!r}"
        outcome, code, stdout = self._data.draw(answers)
        returncode = 0 if outcome is Outcome.OK else code or 1
        return CommandResult(tuple(argv), outcome, returncode, stdout)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data())
def test_collector_never_raises(data: st.DataObject) -> None:
    facts = collect_nvidia(ArbitraryNvidiaSmi(data), which=lambda _: SMI)
    assert isinstance(facts, NvidiaFacts)
    assert facts.smi_path == SMI
    for gpu in facts.gpus:
        assert gpu.name
        assert gpu.compute_capability is None or "." in gpu.compute_capability
