"""Checks of each inspected interpreter: usable, the shell's own, its PyTorch."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from gpu_stack_inspector.models import (
    Confidence,
    Evidence,
    FailureKind,
    Finding,
    InterpreterFacts,
    Severity,
    Snapshot,
    Status,
)
from gpu_stack_inspector.rules.catalog import RULES
from gpu_stack_inspector.rules.engine import RuleContext
from gpu_stack_inspector.rules.torch_build import TorchBuildKind, classify_torch

_CANNOT_START = frozenset({FailureKind.NOT_FOUND, FailureKind.PERMISSION_DENIED})


def _usable(interpreter: InterpreterFacts) -> bool:
    return interpreter.version is not None


@RULES.rule("GSI-ENV-001", "The interpreter can be inspected")
def interpreter_usable(snapshot: Snapshot, context: RuleContext) -> Iterator[Finding]:
    """Whether each named interpreter started and answered the metadata probe.

    "Cannot be started" (a wrong path, a deleted environment, no execute permission) is
    told apart from "starts and crashes" (a broken environment, a damaged standard
    library), because they need different fixes. An interpreter that answered only in
    part passes with the problems listed as limitations.
    """
    for interpreter in snapshot.interpreters:
        details = tuple(f.detail for f in interpreter.failures)
        if _usable(interpreter):
            kind = ", ".join(
                part
                for part in (
                    interpreter.implementation,
                    "conda env" if interpreter.conda else None,
                    "venv" if interpreter.virtualenv else None,
                )
                if part
            )
            yield Finding(
                id="GSI-ENV-001",
                status=Status.PASS,
                severity=Severity.INFO,
                confidence=Confidence.HIGH if not details else Confidence.MEDIUM,
                summary=f"Python {interpreter.version} ({kind or 'python'})",
                reason=f"Inspected {interpreter.path}.",
                subject=interpreter.label,
                evidence=(Evidence("python-probe", "executable", interpreter.executable),),
                limitations=details,
            )
            continue
        cannot_start = any(f.kind in _CANNOT_START for f in interpreter.failures)
        yield Finding(
            id="GSI-ENV-001",
            status=Status.FAIL,
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            summary=(
                "The interpreter cannot be started"
                if cannot_start
                else "The interpreter started but did not answer"
            ),
            reason=details[0] if details else "No answer from the metadata probe.",
            subject=interpreter.label,
            evidence=tuple(Evidence("python-probe", "failure", detail) for detail in details),
            recommended_action=(
                "Check the path: the environment may have been moved or deleted."
                if cannot_start
                else "Run the interpreter by hand to see the whole error; a damaged "
                "environment usually has to be recreated."
            ),
            verify_command=f"{interpreter.path} --version",
        )


def _environment_directory(path: str) -> str:
    # Same directory means same environment: bin/python and bin/python3 of one venv match,
    # two venvs do not. Symlinks are not followed, they would make every venv equal to
    # its base interpreter, which is why this is absolute() and not resolve().
    return str(Path(path).absolute().parent)


@RULES.rule("GSI-ENV-002", "The interpreter is the one the shell runs")
def shell_interpreter(snapshot: Snapshot, context: RuleContext) -> Iterator[Finding]:
    """Whether an inspected interpreter is the python the user's shell would start.

    A common source of confusion: the environment that was checked (or fixed) is not the
    one a plain `python` command runs, for example because another venv or conda env is
    active. Only checked when a single interpreter is inspected: naming several means
    other environments are being compared on purpose. Interpreters count as the same when
    they live in the same directory (symlinks are not followed), so two interpreters of
    one environment in different directories are reported as different.
    """
    if context.shell_python is None or len(snapshot.interpreters) != 1:
        return
    shell_directory = _environment_directory(context.shell_python)
    for interpreter in snapshot.interpreters:
        if not _usable(interpreter):
            continue
        if _environment_directory(interpreter.path) == shell_directory:
            continue
        yield Finding(
            id="GSI-ENV-002",
            status=Status.WARN,
            severity=Severity.LOW,
            confidence=Confidence.MEDIUM,
            summary="This is not the python your shell runs",
            reason=(
                f"Your shell runs {context.shell_python}. Results for {interpreter.path} do not "
                "apply to programs started with a plain `python` command."
            ),
            subject=interpreter.label,
            evidence=(Evidence("PATH", "python", context.shell_python),),
            recommended_action=(
                "Activate this environment before running your program, or start it with "
                "this interpreter's full path."
            ),
            verify_command="which python",
        )


# Installed packages that cannot work without torch.
TORCH_DEPENDENTS = frozenset(
    {"torchvision", "torchaudio", "xformers", "flash-attn", "bitsandbytes", "accelerate",
     "deepspeed", "vllm"}
)  # fmt: skip


@RULES.rule("GSI-TORCH-001", "PyTorch is installed")
def torch_installed(snapshot: Snapshot, context: RuleContext) -> Iterator[Finding]:
    """Whether the interpreter has PyTorch, and whether it needs it.

    An environment without torch is only a problem when it contains packages that need
    it (accelerate, bitsandbytes, torchvision…): then the environment is incomplete.
    Otherwise the PyTorch checks do not apply to it.
    """
    for interpreter in snapshot.interpreters:
        if not _usable(interpreter):
            continue
        if interpreter.torch is not None:
            yield Finding(
                id="GSI-TORCH-001",
                status=Status.PASS,
                severity=Severity.INFO,
                confidence=Confidence.HIGH,
                summary=f"torch {interpreter.torch.version} is installed",
                reason=f"Found in {interpreter.torch.location or 'an unknown location'}.",
                subject=interpreter.label,
                evidence=(Evidence("python-metadata", "torch", interpreter.torch.version),),
            )
            continue
        dependents = sorted(d.name for d in interpreter.distributions if d.name in TORCH_DEPENDENTS)
        if dependents:
            yield Finding(
                id="GSI-TORCH-001",
                status=Status.WARN,
                severity=Severity.MEDIUM,
                confidence=Confidence.HIGH,
                summary="PyTorch is missing, but packages that need it are installed",
                reason=f"{', '.join(dependents)} cannot work without torch.",
                subject=interpreter.label,
                evidence=tuple(
                    Evidence("python-metadata", name, "installed") for name in dependents
                ),
                recommended_action=(
                    "Install PyTorch with the selector at https://pytorch.org/get-started/locally/"
                    ", choosing a CUDA version the driver supports."
                ),
                verify_command=f"gpu-stack-inspector inspect --python {interpreter.path}",
            )
        else:
            yield Finding(
                id="GSI-TORCH-001",
                status=Status.NOT_APPLICABLE,
                severity=Severity.INFO,
                confidence=Confidence.HIGH,
                summary="No PyTorch in this environment",
                reason="Nothing installed here needs it, so the PyTorch checks do not apply.",
                subject=interpreter.label,
            )


_SELECTOR = "https://pytorch.org/get-started/locally/"


@RULES.rule("GSI-TORCH-002", "The PyTorch build supports CUDA", sources=(_SELECTOR,))
def torch_build_supports_cuda(snapshot: Snapshot, context: RuleContext) -> Iterator[Finding]:
    """Whether the installed PyTorch is a CUDA build, judged from static evidence only.

    The evidence is the version tag, torch/version.py, the CUDA libraries in torch/lib,
    the nvidia-* requirements and conda's package records; nothing is imported. A CPU-only
    build where CUDA is expected is the most common reason for
    `torch.cuda.is_available()` returning False. Conflicting evidence gives "unknown",
    never a guess. ROCm builds are named but not judged: this tool checks NVIDIA CUDA.
    """
    max_cuda = snapshot.nvidia.max_cuda_version if snapshot.nvidia else None
    for interpreter in snapshot.interpreters:
        if not _usable(interpreter) or interpreter.torch is None:
            continue
        build = classify_torch(interpreter.torch)
        subject, confidence, evidence = interpreter.label, build.confidence, build.evidence
        if build.kind is TorchBuildKind.CUDA:
            target = f"CUDA {build.cuda_version}" if build.cuda_version else "CUDA"
            yield Finding(
                id="GSI-TORCH-002",
                status=Status.PASS,
                severity=Severity.INFO,
                summary=f"torch {interpreter.torch.version} is a {target} build",
                reason="The metadata of this torch shows a CUDA build. Nothing was imported.",
                limitations=build.conflicts,
                subject=subject,
                confidence=confidence,
                evidence=evidence,
            )
        elif build.kind is TorchBuildKind.CPU:
            choose = (
                f"a CUDA version up to {max_cuda} (what this driver supports)"
                if max_cuda
                else "a CUDA version your NVIDIA driver supports"
            )
            yield Finding(
                id="GSI-TORCH-002",
                status=Status.FAIL if context.cuda_expected else Status.PASS,
                severity=Severity.HIGH if context.cuda_expected else Severity.INFO,
                summary=f"torch {interpreter.torch.version} is a CPU-only build",
                reason=(
                    "This build has no CUDA support, so torch.cuda.is_available() is False "
                    "here whatever the driver can do."
                ),
                recommended_action=(
                    f"Install a CUDA build with the selector at {_SELECTOR}, choosing {choose}."
                ),
                verify_command=f"gpu-stack-inspector inspect --python {interpreter.path}",
                subject=subject,
                confidence=confidence,
                evidence=evidence,
            )
        elif build.kind is TorchBuildKind.ROCM:
            yield Finding(
                id="GSI-TORCH-002",
                status=Status.NOT_APPLICABLE,
                severity=Severity.INFO,
                summary=f"torch {interpreter.torch.version} is a ROCm build",
                reason="ROCm builds target AMD GPUs; this tool only checks NVIDIA CUDA.",
                subject=subject,
                confidence=confidence,
                evidence=evidence,
            )
        else:
            yield Finding(
                id="GSI-TORCH-002",
                status=Status.UNKNOWN,
                severity=Severity.MEDIUM,
                summary="The kind of PyTorch build could not be determined",
                reason=(
                    "; ".join(build.conflicts)
                    if build.conflicts
                    else "No tag, version file, library or package record says which build it is."
                ),
                recommended_action=(
                    "The installation looks inconsistent; reinstalling torch in this "
                    "environment usually fixes it."
                    if build.conflicts
                    else None
                ),
                subject=subject,
                confidence=confidence,
                evidence=evidence,
            )


def _unavailable_cause(snapshot: Snapshot, interpreter: InterpreterFacts) -> str | None:
    """The finding that explains torch.cuda.is_available() being False, if one does."""
    if classify_torch(interpreter.torch).kind is TorchBuildKind.CPU:
        return "GSI-TORCH-002"
    nvidia = snapshot.nvidia
    if nvidia is not None and nvidia.smi_path is not None and nvidia.driver_version is None:
        return "GSI-SYS-002"
    if nvidia is not None and nvidia.driver_version is not None and not nvidia.gpus:
        return "GSI-SYS-003"
    hidden = next((v for v in snapshot.cuda_environment if v.name == "CUDA_VISIBLE_DEVICES"), None)
    if hidden is not None and hidden.value.strip() in {"", "-1"}:
        return "GSI-ENV-003"
    return None


@RULES.rule("GSI-TORCH-003", "CUDA works at runtime")
def cuda_works(snapshot: Snapshot, context: RuleContext) -> Iterator[Finding]:
    """Whether importing torch and running a tiny CUDA operation worked (probe --runtime).

    The only check that proves CUDA works: it imports torch in the target interpreter,
    asks torch.cuda.is_available(), lists the devices and runs a tensor operation on the
    GPU. A failure names the step and the error. "No kernel image is available" means the
    build has no code for this GPU's compute capability. When CUDA is unavailable, the
    finding points at the check that explains why, if one does.
    """
    for interpreter in snapshot.interpreters:
        runtime = interpreter.runtime
        if runtime is None:
            continue
        check = f"gpu-stack-inspector probe --runtime --python {interpreter.path}"
        failed = next((s for s in runtime.steps if not s.ok), None)
        if runtime.failure is not None or (failed is not None and failed.step == "import torch"):
            detail = runtime.failure.detail if runtime.failure else (failed.error if failed else "")
            yield Finding(
                id="GSI-TORCH-003",
                status=Status.FAIL,
                severity=Severity.HIGH,
                confidence=Confidence.HIGH,
                summary=(
                    "import torch fails"
                    if runtime.failure is None
                    else "The interpreter did not survive using CUDA"
                ),
                reason=detail or "No detail was reported.",
                recommended_action=(
                    "Run `python -c \"import torch; print(torch.ones(1, device='cuda'))\"` in this "
                    "environment to see the whole error; missing or mismatched CUDA libraries in "
                    "the environment are the usual cause."
                ),
                evidence=tuple(Evidence("runtime", s.step, s.error or "ok") for s in runtime.steps),
                subject=interpreter.label,
                verify_command=check,
            )
        elif runtime.cuda_available is False:
            yield Finding(
                id="GSI-TORCH-003",
                status=Status.FAIL if context.cuda_expected else Status.NOT_APPLICABLE,
                severity=Severity.HIGH,
                confidence=Confidence.HIGH,
                summary="torch.cuda.is_available() is False",
                reason=f"torch {runtime.torch_version or '?'} found no usable CUDA device.",
                caused_by=_unavailable_cause(snapshot, interpreter),
                evidence=(Evidence("runtime", "cuda available", "False"),),
                subject=interpreter.label,
                verify_command=check,
            )
        elif runtime.tensor_ok is False and failed is not None:
            error = failed.error or ""
            capability = runtime.devices[0].compute_capability if runtime.devices else None
            no_kernels = "no kernel image" in error.lower()
            yield Finding(
                id="GSI-TORCH-003",
                status=Status.FAIL,
                severity=Severity.HIGH,
                confidence=Confidence.HIGH,
                summary="A CUDA operation failed",
                reason=error,
                recommended_action=(
                    f"This torch build has no GPU code for compute capability "
                    f"{capability or 'of this GPU'}; install a build that supports this GPU "
                    "(see the PyTorch release notes)."
                    if no_kernels
                    else "Check the driver with `nvidia-smi`, then run the operation by hand "
                    "to see the whole error."
                ),
                evidence=(Evidence("runtime", failed.step, error),),
                subject=interpreter.label,
                verify_command=check,
            )
        elif runtime.tensor_ok is True:
            device = runtime.devices[0].name if runtime.devices else "the GPU"
            seconds = next((s.seconds for s in runtime.steps if s.step == "tensor operation"), None)
            yield Finding(
                id="GSI-TORCH-003",
                status=Status.PASS,
                severity=Severity.INFO,
                confidence=Confidence.HIGH,
                summary=f"A CUDA operation ran on {device}",
                reason=(
                    f"torch {runtime.torch_version or '?'} with CUDA {runtime.cuda_version or '?'}"
                    + (
                        f"; the tensor operation took {seconds:.2f} s."
                        if seconds is not None
                        else "."
                    )
                ),
                evidence=tuple(Evidence("runtime", s.step, "ok") for s in runtime.steps),
                subject=interpreter.label,
                verify_command=check,
            )
