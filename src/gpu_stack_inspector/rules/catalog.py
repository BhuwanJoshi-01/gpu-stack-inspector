"""The built-in rules. Modules register into RULES when imported, in evaluation order."""

from __future__ import annotations

from gpu_stack_inspector.rules.engine import Registry

RULES = Registry()


def builtin_rules() -> Registry:
    """All built-in rules: host first, then interpreters, then compatibility."""
    from gpu_stack_inspector.rules import host  # noqa: F401 - registers on import

    return RULES
