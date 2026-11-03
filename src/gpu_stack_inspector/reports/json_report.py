"""JSON output: the snapshot as plain data, for scripts, CI and bug reports."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from gpu_stack_inspector.models import InterpreterFacts, Snapshot


def snapshot_to_dict(snapshot: Snapshot) -> dict[str, Any]:
    """Plain dicts, lists and scalars; enums become their documented string values."""
    plain = _plain(snapshot)
    if not isinstance(plain, dict):  # cannot happen: a dataclass becomes a dict
        raise TypeError("snapshot did not serialise to an object")
    return plain


def render_json(snapshot: Snapshot) -> str:
    """Indented JSON with a trailing newline. Pass a redacted snapshot."""
    return json.dumps(snapshot_to_dict(snapshot), indent=2, ensure_ascii=False) + "\n"


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        result = {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
        if isinstance(value, InterpreterFacts):
            result["virtualenv"] = value.virtualenv  # derived, but what readers look for
        return result
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value
