# Roadmap

Requirement IDs (`GPU-001` …) are used in issues, pull requests and commit trailers (`Refs: GPU-003`).
A requirement is done when its tests pass and the behaviour was checked, not when the code is written.

## v0.1: inspect and compare explicitly given interpreters

| ID | Feature | Done when |
| --- | --- | --- |
| GPU-001 | CLI help and version | `gpu-stack-inspector --help` is coherent and nothing needs root |
| GPU-002 | OS, driver and GPU collector | `nvidia-smi` output is parsed when present; no NVIDIA tooling gives `unknown` or `not_applicable`, never a crash |
| GPU-003 | Explicit interpreter inspection | `--python /path/to/python` reports version, executable and selected distributions from inside that interpreter |
| GPU-004 | PyTorch build metadata | Tells missing, CPU build, CUDA build and unknown apart **without importing PyTorch** |
| GPU-005 | Environment comparison | `compare --python A --python B` shows per-field differences in text and JSON |
| GPU-006 | Evidence-backed rules | Every finding has a stable code, evidence, a confidence and a way to verify it |
| GPU-007 | Markdown and JSON reports | Reports redact user names and home paths and never contain full environment dumps |
| GPU-008 | Optional runtime test | `probe --runtime --python A` imports PyTorch, checks CUDA and runs a tiny tensor operation; timeouts and errors are kept as evidence |
| GPU-009 | Offline tests | The unit tests need no NVIDIA hardware and include fixtures from real machines |
| GPU-010 | Packaging | Editable install, wheel install and CLI smoke tests run in CI |

## v0.2

- Discover `.venv` folders and user-selected conda or pyenv environments, from an explicit list of roots (no
  scanning of whole disks).
- Report selected CUDA-related variables (`CUDA_HOME`, `CUDA_PATH`, `LD_LIBRARY_PATH`, `CUDA_VISIBLE_DEVICES`) as
  structured, redacted signals.
- Flag duplicated or competing NVIDIA/CUDA library paths as risks, not as confirmed causes.
- Optional TensorFlow and JAX probes.

## Later

- WSL2 signals; Docker and NVIDIA Container Toolkit state; container vs host GPU visibility.
- Workload checks such as a minimum compute capability.

## Not planned

Driver or toolkit installation and upgrades, uninstalling or reinstalling packages, generated shell commands,
ROCm/Vulkan/Metal diagnosis, model downloads, benchmarks and background monitoring.
