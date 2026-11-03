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
import re
import site
import sys
import sysconfig

PROTOCOL = 1
BEGIN = "GSI-PROBE-BEGIN"
END = "GSI-PROBE-END"

# Distributions that decide whether and how a Python environment uses an NVIDIA GPU.
# Only these are reported: a full package list is noise, and can reveal private projects.
EXACT = frozenset(
    [
        "torch", "torchvision", "torchaudio", "triton", "pytorch-triton", "pytorch-triton-rocm",
        "xformers", "flash-attn", "bitsandbytes", "deepspeed", "vllm", "accelerate",
        "jax", "jaxlib", "tensorflow", "tensorflow-cpu", "tensorflow-gpu", "tf-nightly", "keras",
        "cupy", "onnxruntime", "onnxruntime-gpu", "tensorrt", "numpy",
    ]
)  # fmt: skip
PREFIXES = ("nvidia-", "cupy-cuda", "jax-cuda", "tensorrt-")


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


def normalize(name: str) -> str:
    """PEP 503 normalisation: ``Nvidia_CuBLAS.cu12`` → ``nvidia-cublas-cu12``."""
    return re.sub(r"[-_.]+", "-", name).lower()


def wanted(name: str) -> bool:
    normalized = normalize(name)
    return normalized in EXACT or normalized.startswith(PREFIXES)


def distributions() -> list[dict[str, str]]:
    """Allowlisted distributions on sys.path, read from metadata without importing them."""
    from importlib import metadata

    found = []
    for dist in metadata.distributions():
        name = dist.metadata["Name"] if dist.metadata is not None else None
        version = dist.version
        if not name or not version or not wanted(name):
            continue
        found.append(
            {"name": normalize(name), "version": version, "location": str(dist.locate_file(""))}
        )
    return sorted(found, key=lambda d: (d["name"], d["location"]))


def main() -> None:
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "python": python_facts(),
        "distributions": distributions(),
    }
    sys.stdout.write(BEGIN + json.dumps(result, ensure_ascii=True, sort_keys=True) + END + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
