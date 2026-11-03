"""Comparing dotted version numbers as reported by drivers, nvcc and wheels."""

from __future__ import annotations

import re

_LEADING_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+)*")


def version_tuple(text: str | None) -> tuple[int, ...] | None:
    """``12.8`` → ``(12, 8)``, ``595.99.02`` → ``(595, 99, 2)``; no leading number → None."""
    if not text:
        return None
    match = _LEADING_VERSION.match(text.strip())
    if match is None:
        return None
    return tuple(int(part) for part in match.group(0).split("."))


def major_minor(text: str | None) -> tuple[int, int] | None:
    """The CUDA-relevant part of a version: ``12.4.131`` → ``(12, 4)``, ``13`` → ``(13, 0)``."""
    parts = version_tuple(text)
    if parts is None:
        return None
    return parts[0], parts[1] if len(parts) > 1 else 0
