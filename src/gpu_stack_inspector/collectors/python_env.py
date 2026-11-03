"""Target Python interpreters, inspected from the outside.

The tool never imports a target environment's packages into its own process; it runs
the target interpreter as a subprocess.
"""

from __future__ import annotations

import os
from pathlib import Path

from gpu_stack_inspector.models import FailureKind, ProbeFailure

PROBE = "python"


def interpreter_path(
    raw: str, *, cwd: Path | None = None
) -> tuple[Path | None, ProbeFailure | None]:
    """Turn a ``--python`` argument into an absolute path that can be run.

    ``~`` is expanded and relative paths are taken from ``cwd``. Symlinks are kept on
    purpose: a virtual environment's ``bin/python`` is a symlink to the base
    interpreter, and only the symlink's location tells Python it runs inside the venv.
    Returns the path, or the reason it cannot be used.
    """
    text = raw.strip()
    if not text:
        return None, _failure(FailureKind.NOT_FOUND, "no interpreter path was given")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (cwd or Path.cwd()) / path
    if not path.exists():
        if path.is_symlink():
            target = path.readlink()
            detail = f"{path} is a symlink to {target}, which does not exist"
        else:
            detail = f"{path} does not exist"
        return None, _failure(FailureKind.NOT_FOUND, detail)
    if path.is_dir():
        return None, _failure(FailureKind.NOT_FOUND, f"{path} is a directory, not an interpreter")
    if not os.access(path, os.X_OK):
        return None, _failure(FailureKind.PERMISSION_DENIED, f"{path} is not executable")
    return path, None


def _failure(kind: FailureKind, detail: str) -> ProbeFailure:
    return ProbeFailure(PROBE, kind, detail)
