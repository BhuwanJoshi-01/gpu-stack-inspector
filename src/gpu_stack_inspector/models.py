"""Typed facts and findings shared by collectors, rules and renderers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

SCHEMA_VERSION = 1
"""Version of the snapshot/JSON report layout; bumped only for incompatible changes."""

FINDING_ID = re.compile(r"GSI-[A-Z]+-[0-9]{3}")
"""Finding codes are part of the public contract: ``GSI-<AREA>-<NNN>``, e.g. ``GSI-TORCH-002``."""


class Status(StrEnum):
    """Outcome of one check.

    Only ``PASS`` means that a check ran and succeeded. A check that could not run is
    ``UNKNOWN``; one that was deliberately skipped (an opt-in probe nobody asked for) is
    ``NOT_TESTED``; one that does not apply to this machine is ``NOT_APPLICABLE``.
    """

    PASS = "pass"  # noqa: S105 - a status name, not a password
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    NOT_TESTED = "not_tested"


class Severity(StrEnum):
    """How much a finding matters to the user's goal, independent of certainty."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Confidence(StrEnum):
    """How well the evidence supports a finding."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class Evidence:
    """One observed fact that a finding relies on.

    ``source`` names where the fact came from (``nvidia-smi``, ``python-metadata``…),
    ``key`` what it is (``driver_version``…). ``value`` is ``None`` when the fact is
    missing, which is itself evidence.
    """

    source: str
    key: str
    value: str | None


@dataclass(frozen=True, slots=True)
class Finding:
    """A diagnosis with everything a user needs to judge and verify it.

    ``subject`` says what the finding is about: ``host`` or the label of an inspected
    environment. ``verify_command`` is a command the user can run to check the result
    after acting on ``recommended_action``.
    """

    id: str
    status: Status
    severity: Severity
    confidence: Confidence
    summary: str
    reason: str
    subject: str = "host"
    evidence: tuple[Evidence, ...] = ()
    recommended_action: str | None = None
    verify_command: str | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not FINDING_ID.fullmatch(self.id):
            raise ValueError(f"finding id {self.id!r} does not match GSI-<AREA>-<NNN>")
        if not self.summary.strip():
            raise ValueError(f"finding {self.id} has an empty summary")


class FailureKind(StrEnum):
    """Why a probe produced no usable facts."""

    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    NONZERO_EXIT = "nonzero_exit"
    MALFORMED_OUTPUT = "malformed_output"
    OS_ERROR = "os_error"


@dataclass(frozen=True, slots=True)
class ProbeFailure:
    """A probe that could not deliver its facts.

    Collectors return these instead of raising, so one broken tool never hides the
    results of the others, and rules can report ``unknown`` with the reason.
    """

    probe: str
    kind: FailureKind
    detail: str


def utc_timestamp() -> str:
    """Current time as an ISO 8601 UTC string with second precision."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Everything one run observed and concluded; the input of every renderer.

    Host and interpreter facts join this record as their collectors are added.
    """

    tool_version: str
    created_at: str
    schema_version: int = SCHEMA_VERSION
    probe_failures: tuple[ProbeFailure, ...] = ()
    findings: tuple[Finding, ...] = ()
