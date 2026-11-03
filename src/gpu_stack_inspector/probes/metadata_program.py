"""The metadata probe: run inside a *target* interpreter as ``python -c <this file>``.

Rules for this file, checked by tests:

- Python 3.8 syntax and the standard library only; targets can be much older than the
  tool itself.
- Nothing from the target environment is imported. Package metadata is read from disk.
- Exactly one JSON document is printed, between BEGIN and END, so whatever a target's
  sitecustomize or .pth files print cannot be mistaken for the result.
"""

from __future__ import annotations

import json
import os
import site
import sys
import sysconfig

PROTOCOL = 1
BEGIN = "GSI-PROBE-BEGIN"
END = "GSI-PROBE-END"


def python_facts() -> dict[str, object]:
    enable_user_site = site.ENABLE_USER_SITE
    return {
        "version": "%d.%d.%d" % tuple(sys.version_info[:3]),
        "implementation": sys.implementation.name,
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", sys.prefix),
        "conda": os.path.isdir(os.path.join(sys.prefix, "conda-meta")),
        "platform": sysconfig.get_platform(),
        "user_site": None if enable_user_site is None else bool(enable_user_site),
        "pythonpath_set": bool(os.environ.get("PYTHONPATH")),
    }


def main() -> None:
    result: dict[str, object] = {"protocol": PROTOCOL, "python": python_facts()}
    sys.stdout.write(BEGIN + json.dumps(result, ensure_ascii=True, sort_keys=True) + END + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
