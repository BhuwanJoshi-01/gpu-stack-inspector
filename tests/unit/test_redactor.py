"""What reports may and may not contain."""

from __future__ import annotations

from typing import Any

import pytest

pending = pytest.mark.xfail(strict=True, reason="GPU-007: redactor pending")


def redactor(**context: Any) -> Any:
    from gpu_stack_inspector.privacy import RedactionContext, Redactor

    values = {"home": "/home/alice", "user": "alice", "hostname": "alice-laptop"}
    values.update(context)
    return Redactor(RedactionContext(**values))


def text(value: str, **context: Any) -> str:
    result: str = redactor(**context).text(value)
    return result


# --- home directories ---------------------------------------------------------------


@pending
@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("/home/alice/envs/a/bin/python", "~/envs/a/bin/python"),
        ("/home/alice", "~"),
        ("python at /home/alice/.venv/bin/python3.12.", "python at ~/.venv/bin/python3.12."),
        ("/home/bob/.venv/bin/python", "/home/<user>/.venv/bin/python"),
        ("/home/alice2/x", "/home/<user>/x"),
        ("/Users/carol/miniforge3/bin/python", "/Users/<user>/miniforge3/bin/python"),
        ("C:\\Users\\dave\\envs\\a\\python.exe", "C:\\Users\\<user>\\envs\\a\\python.exe"),
        ("/usr/local/cuda-12.8/bin/nvcc", "/usr/local/cuda-12.8/bin/nvcc"),
        ("/home/<user>/.venv", "/home/<user>/.venv"),
    ],
)
def test_home_directories_are_hidden(raw: str, clean: str) -> None:
    assert text(raw) == clean


# --- user and host names ----------------------------------------------------------------


@pending
@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("owned by alice", "owned by <user>"),
        ("alice's environment", "<user>'s environment"),
        ("malice and alice_env stay", "malice and alice_env stay"),
        ("connected to alice-laptop:22", "connected to <host>:22"),
    ],
)
def test_user_and_host_names_are_hidden_as_whole_words(raw: str, clean: str) -> None:
    assert text(raw) == clean


@pending
def test_short_or_generic_names_are_not_treated_as_secrets() -> None:
    assert text("al is short", user="al", home="/home/al") == "al is short"
    assert text("root cause", user="root", home="/root") == "root cause"
    assert text("Ubuntu 26.04 on ubuntu", hostname="ubuntu") == "Ubuntu 26.04 on ubuntu"
    assert text("localhost:8000", hostname="localhost") == "localhost:8000"


# --- secrets ------------------------------------------------------------------------------


@pending
@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("HF_TOKEN=hf_" + "a" * 34, "HF_TOKEN=<redacted>"),
        ("export OPENAI_API_KEY='sk-abc'", "export OPENAI_API_KEY=<redacted>"),
        ("password: hunter2", "password: <redacted>"),
        ("DB_PASSWORD = s3cr3t rest", "DB_PASSWORD = <redacted> rest"),
        ("AWS_SECRET_ACCESS_KEY=abc/def", "AWS_SECRET_ACCESS_KEY=<redacted>"),
        ('{"api_key": "abc123"}', '{"api_key": <redacted>}'),
        ("accessToken=xyz", "accessToken=<redacted>"),
        ("BASIC_AUTH=user:pass", "BASIC_AUTH=<redacted>"),
    ],
)
def test_secret_assignments_keep_the_name_and_hide_the_value(raw: str, clean: str) -> None:
    assert text(raw) == clean


@pending
@pytest.mark.parametrize(
    "harmless",
    [
        "tokenizers==0.20.1",
        "CUDA_HOME=/usr/local/cuda",
        "author=someone",
        "KEYBOARD=us",
        "Python: 3.11.15",
        "keyring 25.6.0",
    ],
)
def test_ordinary_settings_are_left_alone(harmless: str) -> None:
    assert text(harmless) == harmless


@pending
@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("postgres://alice:S3cr3t@db.local:5432/app", "postgres://<redacted>@db.local:5432/app"),
        ("postgres://u:p@ss@db/app", "postgres://<redacted>@db/app"),
        ("https://token@github.com/org/repo", "https://<redacted>@github.com/org/repo"),
        ("https://pypi.org/simple/torch/", "https://pypi.org/simple/torch/"),
    ],
)
def test_credentials_in_urls_are_hidden(raw: str, clean: str) -> None:
    assert text(raw) == clean


@pending
@pytest.mark.parametrize(
    "token",
    [
        "ghp_" + "A1" * 18,
        "github_pat_" + "a" * 30,
        "hf_" + "b" * 34,
        "sk-proj-" + "c" * 40,
        "sk-ant-api03-" + "d" * 40,
        "AKIA" + "E" * 16,
        "xoxb-1234567890-abcdefghij",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    ],
)
def test_known_token_formats_are_hidden_anywhere(token: str) -> None:
    assert text(f"error: {token} rejected") == "error: <redacted-token> rejected"


@pending
def test_bearer_tokens_are_hidden() -> None:
    assert text("Authorization: Bearer abc.def.ghi123") == "Authorization: Bearer <redacted-token>"


@pending
def test_control_characters_are_stripped_while_redacting() -> None:
    assert text("\x1b[31m/home/alice/x\x1b[0m") == "~/x"


@pending
def test_redaction_is_idempotent() -> None:
    raw = "alice at /home/alice and /home/bob, HF_TOKEN=hf_" + "z" * 34
    once = text(raw)
    assert text(once) == once


# --- whole objects ----------------------------------------------------------------------


@pending
def test_redact_returns_a_redacted_copy_of_nested_dataclasses() -> None:
    from gpu_stack_inspector.models import (
        Confidence,
        Evidence,
        FailureKind,
        Finding,
        ProbeFailure,
        Severity,
        Snapshot,
        Status,
    )

    snapshot = Snapshot(
        tool_version="0.1.0",
        created_at="2026-11-03T10:00:00Z",
        probe_failures=(ProbeFailure("python", FailureKind.TIMEOUT, "/home/alice/b/python hung"),),
        findings=(
            Finding(
                id="GSI-ENV-001",
                status=Status.FAIL,
                severity=Severity.HIGH,
                confidence=Confidence.HIGH,
                summary="Interpreter /home/alice/b/python is not accessible",
                reason="HF_TOKEN=secret was set",
                subject="/home/alice/b",
                evidence=(Evidence("python-probe", "executable", "/home/alice/b/python"),),
            ),
        ),
    )
    clean = redactor().redact(snapshot)
    finding = clean.findings[0]
    assert finding.summary == "Interpreter ~/b/python is not accessible"
    assert finding.reason == "HF_TOKEN=<redacted> was set"
    assert finding.subject == "~/b"
    assert finding.evidence[0].value == "~/b/python"
    assert finding.status is Status.FAIL
    assert clean.probe_failures[0].kind is FailureKind.TIMEOUT
    assert clean.probe_failures[0].detail == "~/b/python hung"
    assert clean.schema_version == 1
    assert snapshot.findings[0].summary.startswith("Interpreter /home/alice")  # original untouched


@pending
def test_redact_handles_plain_containers_and_leaves_other_values_alone() -> None:
    data = {"path": "/home/alice/x", "items": ["/home/alice/y", 3, None, True], "t": ("alice",)}
    assert redactor().redact(data) == {
        "path": "~/x",
        "items": ["~/y", 3, None, True],
        "t": ("<user>",),
    }


@pending
def test_summary_counts_what_was_redacted() -> None:
    r = redactor()
    r.text("/home/alice/a /home/alice/b HF_TOKEN=x postgres://u:p@h/db alice")
    assert r.summary == {"home_path": 2, "secret_value": 1, "url_credentials": 1, "user_name": 1}


@pending
def test_context_for_the_current_user_tolerates_missing_information(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import getpass

    from gpu_stack_inspector.privacy import RedactionContext

    def no_user() -> str:
        raise OSError("no user name")

    monkeypatch.setattr(getpass, "getuser", no_user)
    context = RedactionContext.current()
    assert context.user is None
    assert context.home
