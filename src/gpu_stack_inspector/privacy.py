"""Making collected text safe to print and to share.

Everything that leaves the tool (terminal, JSON, Markdown) is built from redacted data:
redaction runs on the snapshot *before* any renderer sees it.
"""

from __future__ import annotations

import getpass
import re
import socket
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_ESCAPE_SEQUENCE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"  # CSI: colours, cursor movement, screen clearing
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"  # OSC: window titles, hyperlinks
    r"|\x1b[@-_]"  # other two-byte escapes
)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def strip_control(text: str) -> str:
    """Remove terminal escape sequences and control characters, keeping tabs and newlines.

    Collected output can come from a broken or hostile environment; without this a
    crafted package name could rewrite the user's terminal when a report is shown.
    """
    return _CONTROL.sub("", _ESCAPE_SEQUENCE.sub("", text))


@dataclass(frozen=True, slots=True)
class RedactionContext:
    """The identifying values of the machine the report is made on."""

    home: str | None = None
    user: str | None = None
    hostname: str | None = None

    @classmethod
    def current(cls) -> RedactionContext:
        """Read the values for the current user; missing ones stay ``None``."""
        return cls(
            home=_safe(lambda: str(Path.home())),
            user=_safe(getpass.getuser),
            hostname=_safe(socket.gethostname),
        )


def _safe(read: Callable[[], str]) -> str | None:
    try:
        value = read()
    except (OSError, KeyError, RuntimeError):
        return None
    return value or None


Replacement = str | Callable[[re.Match[str]], str]

# Other accounts' home directories. The name may not start with "<", so already redacted
# paths are left alone.
_OTHER_HOMES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(/home/)[^/\s:'\"<>]+"), r"\1<user>"),
    (re.compile(r"(/Users/)[^/\s:'\"<>]+"), r"\1<user>"),
    (re.compile(r"([A-Za-z]:\\Users\\)[^\\\s:'\"<>]+"), r"\1<user>"),
)


class Redactor:
    """Removes identifying and secret values from text and from whole snapshots.

    One redactor is used per report; :attr:`summary` then tells how many values of each
    kind were removed, which the report shows so the user knows redaction happened.
    """

    def __init__(self, context: RedactionContext) -> None:
        self._counts: Counter[str] = Counter()
        self._steps: list[tuple[str, re.Pattern[str], Replacement]] = []
        home = context.home.rstrip("/\\") if context.home else ""
        if len(home) > 1:  # a home of "/" would redact every absolute path
            boundary = r"(?=[/\\]|$|[^\w.-])"
            self._steps.append(("home_path", re.compile(re.escape(home) + boundary), "~"))
        for pattern, replacement in _OTHER_HOMES:
            self._steps.append(("home_path", pattern, replacement))

    def text(self, value: str) -> str:
        """Return ``value`` without control characters, identifying or secret values."""
        result = strip_control(value)
        for category, pattern, replacement in self._steps:
            result, count = pattern.subn(replacement, result)
            self._counts[category] += count
        return result
