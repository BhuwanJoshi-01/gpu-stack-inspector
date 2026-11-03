"""Typed facts and findings shared by collectors, rules and renderers."""

from __future__ import annotations

from enum import StrEnum


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
