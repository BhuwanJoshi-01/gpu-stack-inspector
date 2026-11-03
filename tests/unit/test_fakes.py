import pytest

from gpu_stack_inspector.runner import CommandResult, CommandRunner, Outcome
from tests.fakes import FakeRunner, UnexpectedCommandError


def test_fake_runner_answers_scripted_commands_and_records_calls() -> None:
    fake = FakeRunner()
    fake.script("nvidia-smi", "-L", stdout="GPU 0: NVIDIA GeForce RTX 3050\n")
    runner: CommandRunner = fake
    result = runner.run(["nvidia-smi", "-L"], timeout=8, env_overrides={"LC_ALL": "C"})
    assert result.ok
    assert result.stdout.startswith("GPU 0")
    assert fake.calls[0].argv == ("nvidia-smi", "-L")
    assert fake.calls[0].env_overrides == {"LC_ALL": "C"}


def test_fake_runner_fails_the_test_on_unscripted_commands() -> None:
    with pytest.raises(UnexpectedCommandError, match="rm"):
        FakeRunner().run(["rm", "-rf", "build"], timeout=1)


def test_fake_runner_can_return_any_outcome() -> None:
    fake = FakeRunner()
    fake.script_result(CommandResult(("nvidia-smi",), Outcome.TIMEOUT, duration=8))
    assert fake.run(["nvidia-smi"], timeout=8).outcome is Outcome.TIMEOUT
