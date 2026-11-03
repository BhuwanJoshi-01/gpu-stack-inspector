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
from typing import Callable

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
MAX_METADATA_BYTES = 64 * 1024

# Libraries in torch/lib that only GPU builds ship (Linux, Windows, macOS names).
CUDA_LIBRARIES = ("libtorch_cuda.so", "torch_cuda.dll", "libtorch_cuda.dylib")
HIP_LIBRARIES = ("libtorch_hip.so", "torch_hip.dll")

# torch/version.py lines, old and new style: "cuda = '11.3'", "cuda: Optional[str] = '12.8'",
# "hip: Optional[str] = None". The file is read as text, never imported.
VERSION_PY_LINE = re.compile(
    r"^(?P<key>cuda|hip)\s*(?::[^=\n]*)?=\s*(?:'(?P<single>[^'\n]*)'|\"(?P<double>[^\"\n]*)\"|None)\s*$",
    re.MULTILINE,
)


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
    try:
        from importlib import metadata
    except ImportError:  # Python before 3.8
        return scan_dist_info(sys.path)

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


def torch_facts() -> dict[str, object] | None:
    """Static signals about the torch that ``import torch`` would find, without importing it.

    Returns None when torch is not installed.
    """
    try:
        from importlib import metadata
    except ImportError:  # Python before 3.8: the distributions section still lists torch
        return None
    try:
        dist = metadata.distribution("torch")
    except metadata.PackageNotFoundError:
        return None
    requirements = []
    for requirement in dist.requires or []:
        match = re.match(r"[A-Za-z0-9._-]+", requirement)
        if match and normalize(match.group(0)).startswith("nvidia-"):
            requirements.append(normalize(match.group(0)))
    package = str(dist.locate_file("torch"))
    return {
        "version": dist.version,
        "location": str(dist.locate_file("")),
        "requires_nvidia": sorted(set(requirements)),
        "libraries": torch_libraries(os.path.join(package, "lib")),
        "version_py": torch_version_py(os.path.join(package, "version.py")),
        "conda_records": conda_records(sys.prefix),
    }


def conda_records(prefix: str) -> list[str] | None:
    """Conda packages that decide the torch build, by file name in conda-meta.

    Names like ``pytorch-1.12.0-py3.8_cuda11.3_cudnn8.3.2_0`` carry the build string, so
    the (multi-megabyte) records are never opened. None outside conda environments.
    """
    try:
        names = os.listdir(os.path.join(prefix, "conda-meta"))
    except OSError:
        return None
    wanted_prefixes = ("pytorch-", "cudatoolkit-", "cuda-version-", "cuda-runtime-")
    return sorted(
        name[: -len(".json")]
        for name in names
        if name.endswith(".json") and name.startswith(wanted_prefixes)
    )


def torch_version_py(path: str) -> dict[str, str | None] | None:
    """The cuda and hip versions torch was built with, read from torch/version.py as text.

    None when the file is missing or names neither (a damaged or unusual install).
    """
    try:
        with open(path, "rb") as handle:
            text = handle.read(MAX_METADATA_BYTES).decode("utf-8", "replace")
    except OSError:
        return None
    found: dict[str, str | None] = {}
    for match in VERSION_PY_LINE.finditer(text):
        value = match.group("single")
        if value is None:
            value = match.group("double")
        found.setdefault(match.group("key"), value)
    if not found:
        return None
    return {"cuda": found.get("cuda"), "hip": found.get("hip")}


def torch_libraries(lib_dir: str) -> dict[str, bool] | None:
    """Whether torch/lib holds CUDA or ROCm libraries; None if the folder cannot be read."""
    try:
        names = set(os.listdir(lib_dir))
    except OSError:
        return None
    return {
        "cuda": any(name in names for name in CUDA_LIBRARIES),
        "hip": any(name in names for name in HIP_LIBRARIES),
    }


def scan_dist_info(paths: list[str]) -> list[dict[str, str]]:
    """Fallback without importlib.metadata: read the headers of ``*.dist-info/METADATA``."""
    found = []
    for entry in paths:
        try:
            folders = sorted(os.listdir(entry or "."))
        except OSError:
            continue
        for folder in folders:
            if not folder.endswith(".dist-info"):
                continue
            headers = read_headers(os.path.join(entry, folder, "METADATA"))
            name, version = headers.get("name"), headers.get("version")
            if name and version and wanted(name):
                found.append({"name": normalize(name), "version": version, "location": entry})
    return found


def read_headers(path: str) -> dict[str, str]:
    """``Name:``/``Version:`` style headers up to the first blank line."""
    headers: dict[str, str] = {}
    try:
        with open(path, "rb") as handle:
            text = handle.read(MAX_METADATA_BYTES).decode("utf-8", "replace")
    except OSError:
        return headers
    for line in text.splitlines():
        if not line.strip():
            break
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() not in headers:
            headers[key.strip().lower()] = value.strip()
    return headers


def collect(sections: list[tuple[str, Callable[[], object]]]) -> dict[str, object]:
    """Run each section; an exception is reported and the other sections still run."""
    errors: list[str] = []
    result: dict[str, object] = {"protocol": PROTOCOL, "errors": errors}
    for key, function in sections:
        try:
            result[key] = function()
        except Exception as exc:  # a partial answer is worth more than none
            result[key] = None
            errors.append("%s: %s: %s" % (key, type(exc).__name__, exc))
    return result


def main() -> None:
    result = collect(
        [("python", python_facts), ("distributions", distributions), ("torch", torch_facts)]
    )
    sys.stdout.write(BEGIN + json.dumps(result, ensure_ascii=True, sort_keys=True) + END + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
