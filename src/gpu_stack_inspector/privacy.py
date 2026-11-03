"""Making collected text safe to print and to share.

Everything that leaves the tool (terminal, JSON, Markdown) is built from redacted data:
redaction runs on the snapshot *before* any renderer sees it.
"""

from __future__ import annotations

import re

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
