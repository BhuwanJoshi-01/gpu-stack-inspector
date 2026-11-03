"""The built-in rules. Modules register into RULES when imported, in evaluation order."""

from __future__ import annotations

import importlib

from gpu_stack_inspector.rules.engine import Registry

RULES = Registry()

_MODULES = ("host", "interpreters", "compatibility", "differences")
"""Evaluation order: the machine first, then each interpreter, then how they fit together,
then how the interpreters differ from each other.
Kept as data on purpose: import sorting would otherwise reorder the rules."""


def builtin_rules() -> Registry:
    """All built-in rules, in evaluation order."""
    for name in _MODULES:
        importlib.import_module(f"gpu_stack_inspector.rules.{name}")
    return RULES
