# Changelog

All notable changes to this project are listed here. Versions follow
[Semantic Versioning](https://semver.org/); the JSON report format has its own
`schema_version` and changes only with a new major version of the schema.

## Unreleased

Nothing released yet. The first release (0.1.0) will inspect and compare
explicitly given Python interpreters.

- `gpu-stack-inspector inspect [--python PATH]… [--format text|json]` reports
  the OS, the NVIDIA driver (and the newest CUDA it supports), the GPUs, CUDA
  toolkits and nvcc, allowlisted CUDA variables, and for each interpreter its
  Python version, whether it is a conda env or venv, and its GPU-relevant
  packages, read from metadata without importing them. Without `--python`, the
  `python` on PATH is inspected. Home paths, user and host names and secrets are
  redacted from all output. Exit code 3 when a named interpreter cannot be
  started.
- For each interpreter, inspect says which PyTorch build it has (CUDA with the
  CUDA version it targets, CPU-only, ROCm, or unknown when the evidence
  disagrees), without importing torch. The evidence: the version tag,
  `torch/version.py` read as text, the CUDA libraries in `torch/lib`, the
  nvidia-* requirements, and conda's package records. JSON output has it as
  `torch_build`.
- `inspect` runs 11 checks and lists their findings per host and interpreter:
  driver tools, driver, GPUs, `CUDA_VISIBLE_DEVICES` (GSI-SYS-001…003,
  GSI-ENV-003), each interpreter and whether it is the shell's python
  (GSI-ENV-001, 002), PyTorch installed and a CUDA build (GSI-TORCH-001, 002),
  and nvcc and the driver against the CUDA versions in use (GSI-CUDA-001…003,
  with dated, source-tagged NVIDIA compatibility data). Problems say what to do
  next and how to check it; a consequence of another finding says which one it
  follows from. Findings are in the JSON output too.
- `rules list` and `rules explain CODE` document every check.
- `gpu-stack-inspector --help` and `--version`. Exit codes: 0 when the
  inspection ran (whatever it found), 1 for `--strict` failures, 2 for usage
  errors and internal errors, 3 when required context is missing, 130 for
  Ctrl+C. Internal errors print one redacted line; `GSI_DEBUG=1` shows the
  traceback.
