"""The runtime probe: run inside a *target* interpreter only with --runtime (explicit consent).

Unlike the metadata probe, this imports torch, creates a CUDA context and runs a tiny
operation, because only that proves CUDA works. Rules for this file, checked by tests:

- Python 3.8 syntax and the standard library plus torch; nothing else is imported.
- The workload is tiny (16 floats) and bounded; nothing is written to disk.
- Each step reports its own failure, so the report can say exactly where CUDA broke.
- Exactly one JSON document is printed between BEGIN and END.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Callable

PROTOCOL = 1
BEGIN = "GSI-RUNTIME-BEGIN"
END = "GSI-RUNTIME-END"
STEP = "GSI-RUNTIME-STEP"
"""Printed (and flushed) before each step: if CUDA initialisation kills the interpreter,
no JSON is printed, and the last marker says which step it died in."""
ELEMENTS = 16


def first_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return "%s: %s" % (type(exc).__name__, text[0] if text else "")


def run_steps() -> dict[str, object]:
    result: dict[str, object] = {"protocol": PROTOCOL, "steps": []}
    steps: list[dict[str, object]] = []
    result["steps"] = steps
    state: dict[str, Any] = {}

    def step(name: str, action: Callable[[], object]) -> bool:
        sys.stdout.write("%s %s\n" % (STEP, name))
        sys.stdout.flush()
        started = time.monotonic()
        try:
            value = action()
        except BaseException as exc:  # report where it broke, then stop
            seconds = round(time.monotonic() - started, 3)
            steps.append({"step": name, "ok": False, "error": first_line(exc), "seconds": seconds})
            return False
        seconds = round(time.monotonic() - started, 3)
        steps.append({"step": name, "ok": True, "value": value, "seconds": seconds})
        return True

    def import_torch() -> object:
        import torch  # type: ignore[import-not-found]  # lives in the target, never in the tool

        state["torch"] = torch
        return {"version": torch.__version__, "cuda": torch.version.cuda}

    def cuda_available() -> object:
        torch = state["torch"]
        return bool(torch.cuda.is_available())

    def devices() -> object:
        torch = state["torch"]
        count = int(torch.cuda.device_count())
        found = []
        for index in range(count):
            major, minor = torch.cuda.get_device_capability(index)
            name = str(torch.cuda.get_device_name(index))
            found.append({"index": index, "name": name, "capability": "%d.%d" % (major, minor)})
        return found

    def tensor_operation() -> object:
        torch = state["torch"]
        x = torch.ones(ELEMENTS, device="cuda")
        total = (x * 2).sum()
        torch.cuda.synchronize()
        value = float(total.cpu().item())
        if value != 2.0 * ELEMENTS:
            raise ArithmeticError("expected %s, got %s" % (2.0 * ELEMENTS, value))
        return value

    if not step("import torch", import_torch):
        return result
    if not step("cuda available", cuda_available):
        return result
    if steps[-1]["value"] is not True:
        return result  # nothing more to try: CUDA is not usable from this torch
    if step("devices", devices):
        step("tensor operation", tensor_operation)
    return result


def main() -> None:
    result = run_steps()
    sys.stdout.write(BEGIN + json.dumps(result, ensure_ascii=True, sort_keys=True) + END + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
