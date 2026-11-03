from gpu_stack_inspector.runner import CommandResult, Outcome


def test_only_ok_outcome_counts_as_ok() -> None:
    assert CommandResult(("true",), Outcome.OK, returncode=0).ok
    for outcome in Outcome:
        if outcome is not Outcome.OK:
            assert not CommandResult(("x",), outcome).ok
