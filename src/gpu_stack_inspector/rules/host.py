"""Checks of the machine: driver tools, the driver, the GPUs, variables that hide GPUs."""

from __future__ import annotations

from collections.abc import Iterator

from gpu_stack_inspector.models import (
    Confidence,
    Evidence,
    Finding,
    Severity,
    Snapshot,
    Status,
)
from gpu_stack_inspector.rules.catalog import RULES
from gpu_stack_inspector.rules.engine import RuleContext

SMI = "nvidia-smi"


@RULES.rule("GSI-SYS-001", "NVIDIA driver tools are installed")
def driver_tools(snapshot: Snapshot, context: RuleContext) -> Iterator[Finding]:
    """Whether nvidia-smi, which every NVIDIA driver installs, is on PATH.

    Without it the driver and the GPUs cannot be checked. Its absence usually means no
    NVIDIA driver is installed, but some container images run GPU workloads without it,
    so this is a warning, not a failure. When CUDA is not expected it does not apply.
    """
    nvidia = snapshot.nvidia
    if nvidia is None:
        return
    if nvidia.smi_path is not None:
        yield Finding(
            id="GSI-SYS-001",
            status=Status.PASS,
            severity=Severity.INFO,
            confidence=Confidence.HIGH,
            summary="nvidia-smi is installed",
            reason=f"Found at {nvidia.smi_path}.",
            evidence=(Evidence(SMI, "path", nvidia.smi_path),),
        )
        return
    yield Finding(
        id="GSI-SYS-001",
        status=Status.WARN if context.cuda_expected else Status.NOT_APPLICABLE,
        severity=Severity.MEDIUM if context.cuda_expected else Severity.INFO,
        confidence=Confidence.MEDIUM,
        summary="nvidia-smi was not found",
        reason=(
            "Without NVIDIA's driver tools on PATH the driver and the GPUs cannot be "
            "checked. A system with a working NVIDIA driver normally provides nvidia-smi."
        ),
        evidence=(Evidence(SMI, "path", None),),
        recommended_action=(
            "Install the NVIDIA driver for your distribution (it provides nvidia-smi), or "
            "add its directory to PATH. On WSL it is in /usr/lib/wsl/lib."
        ),
        verify_command="nvidia-smi",
        limitations=("Some container images run GPU workloads without nvidia-smi installed.",),
    )


@RULES.rule("GSI-SYS-002", "The NVIDIA driver answers")
def driver_answers(snapshot: Snapshot, context: RuleContext) -> Iterator[Finding]:
    """Whether nvidia-smi could talk to the NVIDIA driver and report its version.

    nvidia-smi installed but failing usually means the kernel module is not loaded: after
    a driver update without a reboot, when Secure Boot blocks an unsigned module, or when
    the installed module does not match the running kernel. The failure message from
    nvidia-smi is quoted as evidence.
    """
    nvidia = snapshot.nvidia
    if nvidia is None:
        return
    if nvidia.smi_path is None:
        yield Finding(
            id="GSI-SYS-002",
            status=Status.NOT_TESTED,
            severity=Severity.INFO,
            confidence=Confidence.HIGH,
            summary="The NVIDIA driver was not checked",
            reason="nvidia-smi is not installed, so there was nothing to ask.",
            caused_by="GSI-SYS-001",
        )
        return
    if nvidia.driver_version is None:
        details = [f.detail for f in nvidia.failures] or ["nvidia-smi reported no driver version"]
        yield Finding(
            id="GSI-SYS-002",
            status=Status.FAIL if context.cuda_expected else Status.WARN,
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            summary="nvidia-smi could not get an answer from the NVIDIA driver",
            reason=details[0],
            evidence=tuple(Evidence(SMI, "failure", detail) for detail in details),
            recommended_action=(
                "Check that the NVIDIA kernel module is loaded (`lsmod | grep nvidia`). After "
                "a driver update a reboot is usually needed; with Secure Boot an unsigned "
                "module is refused."
            ),
            verify_command="nvidia-smi",
        )
        return
    max_cuda = nvidia.max_cuda_version or "an unknown version"
    yield Finding(
        id="GSI-SYS-002",
        status=Status.PASS,
        severity=Severity.INFO,
        confidence=Confidence.HIGH,
        summary=f"NVIDIA driver {nvidia.driver_version} answers",
        reason=(
            f"The driver supports CUDA up to {max_cuda}. That is the newest CUDA it can run, "
            "not a CUDA toolkit installed on this machine."
        ),
        evidence=(
            Evidence(SMI, "driver_version", nvidia.driver_version),
            Evidence(SMI, "max_cuda_version", nvidia.max_cuda_version),
        ),
        limitations=tuple(f.detail for f in nvidia.failures),
    )


@RULES.rule("GSI-SYS-003", "The driver sees a GPU")
def gpu_visible(snapshot: Snapshot, context: RuleContext) -> Iterator[Finding]:
    """Whether the NVIDIA driver lists at least one GPU.

    A driver that answers but lists no GPU means the GPU is switched off or not handed to
    this system: a hybrid-graphics mode that powers the NVIDIA GPU down, a BIOS setting,
    or a VM or container the GPU was not passed through to.
    """
    nvidia = snapshot.nvidia
    if nvidia is None:
        return
    if nvidia.smi_path is None or nvidia.driver_version is None:
        cause = "GSI-SYS-001" if nvidia.smi_path is None else "GSI-SYS-002"
        yield Finding(
            id="GSI-SYS-003",
            status=Status.NOT_TESTED,
            severity=Severity.INFO,
            confidence=Confidence.HIGH,
            summary="GPUs were not listed",
            reason="The driver could not be asked.",
            caused_by=cause,
        )
        return
    if not nvidia.gpus:
        yield Finding(
            id="GSI-SYS-003",
            status=Status.FAIL if context.cuda_expected else Status.NOT_APPLICABLE,
            severity=Severity.HIGH if context.cuda_expected else Severity.INFO,
            confidence=Confidence.HIGH,
            summary="The NVIDIA driver answers but sees no GPU",
            reason=f"Driver {nvidia.driver_version} reported no devices.",
            evidence=(Evidence(SMI, "gpus", "0"),),
            recommended_action=(
                "Make sure the NVIDIA GPU is switched on for this system: the hybrid-graphics "
                "mode (for example prime-select or supergfxctl), the BIOS setting, or the "
                "GPU passthrough of your VM or container."
            ),
            verify_command="nvidia-smi -L",
        )
        return
    names = ", ".join(
        f"{gpu.name} (compute {gpu.compute_capability or 'unknown'})" for gpu in nvidia.gpus
    )
    count = len(nvidia.gpus)
    yield Finding(
        id="GSI-SYS-003",
        status=Status.PASS,
        severity=Severity.INFO,
        confidence=Confidence.HIGH,
        summary=f"{count} GPU{'s' if count != 1 else ''} visible to the driver",
        reason=names,
        evidence=tuple(Evidence(SMI, f"gpu{gpu.index}", gpu.name) for gpu in nvidia.gpus),
    )


_HIDE_ALL = frozenset({"", "-1", "none", "nodevfiles"})


@RULES.rule(
    "GSI-ENV-003",
    "CUDA_VISIBLE_DEVICES does not hide the GPUs",
    sources=("https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars",),
)
def visible_devices(snapshot: Snapshot, context: RuleContext) -> Iterator[Finding]:
    """Whether CUDA_VISIBLE_DEVICES, when set, leaves the GPUs visible to CUDA programs.

    An empty value (or -1) hides every GPU from CUDA while nvidia-smi still lists it,
    which looks exactly like a broken install. CUDA also stops at the first index that
    does not exist: with "1,0" on a one-GPU machine no GPU is visible at all. GPU UUIDs
    are not checked, because the tool does not collect them.
    """
    variable = next(
        (v for v in snapshot.cuda_environment if v.name == "CUDA_VISIBLE_DEVICES"), None
    )
    if variable is None:
        return
    value = variable.value.strip()
    evidence = (Evidence("environment", "CUDA_VISIBLE_DEVICES", variable.value),)
    shown = repr(variable.value)
    if value.lower() in _HIDE_ALL:
        yield Finding(
            id="GSI-ENV-003",
            status=Status.WARN,
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            summary="CUDA_VISIBLE_DEVICES hides every GPU",
            reason=(
                f"It is set to {shown}, so CUDA programs started from this environment see no "
                "GPU, even though nvidia-smi lists them."
            ),
            evidence=evidence,
            recommended_action=(
                "Unset CUDA_VISIBLE_DEVICES, or set it to the GPU index to use (e.g. 0)."
            ),
            verify_command='echo "$CUDA_VISIBLE_DEVICES"',
        )
        return
    entries = [entry.strip() for entry in value.split(",")]
    count = len(snapshot.nvidia.gpus) if snapshot.nvidia else 0
    if count and all(entry.isdigit() for entry in entries):
        invalid = next((int(e) for e in entries if int(e) >= count), None)
        if invalid is not None:
            usable = entries[: entries.index(str(invalid))]
            yield Finding(
                id="GSI-ENV-003",
                status=Status.WARN,
                severity=Severity.HIGH if not usable else Severity.MEDIUM,
                confidence=Confidence.HIGH,
                summary=f"CUDA_VISIBLE_DEVICES names GPU {invalid}, which does not exist",
                reason=(
                    f"It is set to {shown}, but the driver lists {count} GPU(s). CUDA stops at "
                    "the first index that does not exist, so "
                    + (
                        f"only GPU(s) {', '.join(usable)} are visible."
                        if usable
                        else "no GPU is visible."
                    )
                ),
                evidence=evidence,
                recommended_action="Use indices from `nvidia-smi -L`, or unset the variable.",
                verify_command="nvidia-smi -L",
            )
            return
    yield Finding(
        id="GSI-ENV-003",
        status=Status.PASS,
        severity=Severity.INFO,
        confidence=Confidence.MEDIUM if not all(e.isdigit() for e in entries) else Confidence.HIGH,
        summary=f"CUDA_VISIBLE_DEVICES limits CUDA to {value}",
        reason="CUDA programs only see the GPUs listed there.",
        evidence=evidence,
        limitations=(
            ("GPU UUIDs and MIG devices are not checked: the tool does not collect them.",)
            if not all(e.isdigit() for e in entries)
            else ()
        ),
    )
