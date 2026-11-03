"""Typed facts and findings shared by collectors, rules and renderers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

SCHEMA_VERSION = 1
"""Version of the snapshot/JSON report layout; bumped only for incompatible changes."""

FINDING_ID = re.compile(r"GSI-[A-Z]+-[0-9]{3}")
"""Finding codes are part of the public contract: ``GSI-<AREA>-<NNN>``, e.g. ``GSI-TORCH-002``."""


class Status(StrEnum):
    """Outcome of one check.

    Only ``PASS`` means that a check ran and succeeded. A check that could not run is
    ``UNKNOWN``; one that was deliberately skipped (an opt-in probe nobody asked for) is
    ``NOT_TESTED``; one that does not apply to this machine is ``NOT_APPLICABLE``.
    """

    PASS = "pass"  # noqa: S105 - a status name, not a password
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    NOT_TESTED = "not_tested"


class Severity(StrEnum):
    """How much a finding matters to the user's goal, independent of certainty."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Confidence(StrEnum):
    """How well the evidence supports a finding."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class Evidence:
    """One observed fact that a finding relies on.

    ``source`` names where the fact came from (``nvidia-smi``, ``python-metadata``…),
    ``key`` what it is (``driver_version``…). ``value`` is ``None`` when the fact is
    missing, which is itself evidence.
    """

    source: str
    key: str
    value: str | None


@dataclass(frozen=True, slots=True)
class Finding:
    """A diagnosis with everything a user needs to judge and verify it.

    ``subject`` says what the finding is about: ``host`` or the label of an inspected
    environment. ``verify_command`` is a command the user can run to check the result
    after acting on ``recommended_action``.
    """

    id: str
    status: Status
    severity: Severity
    confidence: Confidence
    summary: str
    reason: str
    subject: str = "host"
    evidence: tuple[Evidence, ...] = ()
    recommended_action: str | None = None
    verify_command: str | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not FINDING_ID.fullmatch(self.id):
            raise ValueError(f"finding id {self.id!r} does not match GSI-<AREA>-<NNN>")
        if not self.summary.strip():
            raise ValueError(f"finding {self.id} has an empty summary")


class FailureKind(StrEnum):
    """Why a probe produced no usable facts."""

    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    NONZERO_EXIT = "nonzero_exit"
    MALFORMED_OUTPUT = "malformed_output"
    OS_ERROR = "os_error"
    PROBE_ERROR = "probe_error"
    """The probe ran, but reported that part of its own work failed."""


@dataclass(frozen=True, slots=True)
class ProbeFailure:
    """A probe that could not deliver its facts.

    Collectors return these instead of raising, so one broken tool never hides the
    results of the others, and rules can report ``unknown`` with the reason.
    """

    probe: str
    kind: FailureKind
    detail: str


def utc_timestamp() -> str:
    """Current time as an ISO 8601 UTC string with second precision."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Everything one run observed and concluded; the input of every renderer.

    ``probe_failures`` collects the failures of every collector in one place, so a report
    can list what could not be checked.
    """

    tool_version: str
    created_at: str
    schema_version: int = SCHEMA_VERSION
    host: HostFacts | None = None
    nvidia: NvidiaFacts | None = None
    toolkit: ToolkitFacts | None = None
    cuda_environment: tuple[EnvironmentVariable, ...] = ()
    interpreters: tuple[InterpreterFacts, ...] = ()
    probe_failures: tuple[ProbeFailure, ...] = ()
    findings: tuple[Finding, ...] = ()


@dataclass(frozen=True, slots=True)
class HostFacts:
    """The operating system the inspection ran on."""

    system: str
    """``platform.system()``: ``Linux``, ``Windows``, ``Darwin``…"""
    machine: str
    """CPU architecture as the OS reports it: ``x86_64``, ``AMD64``, ``aarch64``…"""
    kernel: str | None
    distribution: str | None = None
    """``PRETTY_NAME`` from os-release on Linux, e.g. ``Ubuntu 26.04.1 LTS``."""
    wsl: bool = False
    """Running inside Windows Subsystem for Linux (a fact only; WSL is not validated yet)."""


@dataclass(frozen=True, slots=True)
class GpuDevice:
    """One GPU as nvidia-smi lists it. Values it reports as N/A are ``None``."""

    index: int
    name: str
    memory_total_mib: int | None = None
    compute_capability: str | None = None
    """``major.minor``, e.g. ``8.6``; decides which CUDA kernels a wheel can run."""


@dataclass(frozen=True, slots=True)
class NvidiaFacts:
    """What nvidia-smi reported.

    ``smi_path`` is ``None`` when nvidia-smi is not installed (or not on ``PATH``), which
    is a fact, not a failure. ``failures`` say why other fields are missing.
    """

    smi_path: str | None
    driver_version: str | None = None
    max_cuda_version: str | None = None
    """The newest CUDA version the driver supports (nvidia-smi's "CUDA Version").

    It is not the version of any installed CUDA toolkit or of PyTorch's CUDA runtime.
    """
    gpus: tuple[GpuDevice, ...] = ()
    failures: tuple[ProbeFailure, ...] = ()


@dataclass(frozen=True, slots=True)
class CudaToolkit:
    """A CUDA toolkit directory found on disk.

    ``linked_from`` lists symlinks that point at it, e.g. ``/usr/local/cuda`` for the
    default toolkit. ``version_source`` names the file the version was read from.
    """

    path: str
    version: str | None = None
    version_source: str | None = None
    linked_from: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolkitFacts:
    """Locally installed CUDA toolkits and the ``nvcc`` on ``PATH``.

    Prebuilt PyTorch wheels bring their own CUDA runtime, so a missing toolkit is not a
    problem by itself; it matters for compiling extensions.
    """

    nvcc_path: str | None = None
    nvcc_version: str | None = None
    toolkits: tuple[CudaToolkit, ...] = ()
    failures: tuple[ProbeFailure, ...] = ()


@dataclass(frozen=True, slots=True)
class EnvironmentVariable:
    """An allowlisted variable that was set. An empty value is kept: for
    ``CUDA_VISIBLE_DEVICES`` it means "no GPU is visible"."""

    name: str
    value: str


@dataclass(frozen=True, slots=True)
class Distribution:
    """An installed distribution relevant to GPU use, as the target interpreter sees it.

    ``location`` is the directory it was found in; the same name in two locations (a
    user site and the environment, say) is how shadowed installs show up.
    """

    name: str
    version: str
    location: str | None = None


@dataclass(frozen=True, slots=True)
class InterpreterFacts:
    """What one target Python interpreter reported about itself.

    ``path`` is the interpreter as the user named it, made absolute but with symlinks
    kept: a virtual environment's ``bin/python`` is a symlink, and following it would
    start the base interpreter outside the environment.
    """

    label: str
    path: str
    executable: str | None = None
    version: str | None = None
    implementation: str | None = None
    prefix: str | None = None
    base_prefix: str | None = None
    conda: bool | None = None
    platform: str | None = None
    user_site: bool | None = None
    pythonpath_set: bool | None = None
    distributions: tuple[Distribution, ...] = ()
    failures: tuple[ProbeFailure, ...] = ()

    @property
    def virtualenv(self) -> bool | None:
        """Whether the interpreter runs inside a venv/virtualenv, if that is known."""
        if self.prefix is None or self.base_prefix is None:
            return None
        return self.prefix != self.base_prefix
