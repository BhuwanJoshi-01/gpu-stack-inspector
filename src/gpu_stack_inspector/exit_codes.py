"""Exit codes are a contract: scripts and CI jobs depend on them."""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    """The inspection ran. Findings, even failing ones, do not change this."""

    STRICT_FAILURE = 1
    """``--strict`` was given and a finding reached the configured fail level."""

    ERROR = 2
    """Wrong usage, or the tool itself failed."""

    CONTEXT_UNAVAILABLE = 3
    """Something the command needs is missing, e.g. an interpreter that was named."""

    INTERRUPTED = 130
    """Stopped with Ctrl+C (128 + SIGINT, as shells report it)."""
