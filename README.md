# GPU Stack Inspector

**Explain why your AI code uses the GPU in one Python environment but not another.**

[![CI](https://github.com/BhuwanJoshi-01/gpu-stack-inspector/actions/workflows/ci.yml/badge.svg)](https://github.com/BhuwanJoshi-01/gpu-stack-inspector/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/BhuwanJoshi-01/gpu-stack-inspector/blob/main/LICENSE)
[![Website](https://img.shields.io/badge/website-gpu--stack--inspector-17202e.svg)](https://bhuwanjoshi-01.github.io/gpu-stack-inspector/)

`nvidia-smi` shows your GPU, yet `torch.cuda.is_available()` is `True` in one environment and `False` in another.
GPU Stack Inspector inspects several Python interpreters from one command, shows what differs between them, and
explains which difference matters, with the evidence:

![compare on a laptop with an RTX 3050: a conda environment with torch 2.11.0+cu128 against a venv with torch 2.15.0+cpu. The table shows the torch build CUDA 12.8 versus CPU only and fifteen nvidia CUDA wheels missing from the venv; the findings fail GSI-TORCH-002 for the CPU-only build and warn GSI-DIFF-001 that the environments have different PyTorch builds.](https://raw.githubusercontent.com/BhuwanJoshi-01/gpu-stack-inspector/main/docs/images/compare.svg)

> **Status:** 0.1.0 is the first release, validated on one laptop with five real environments. Reports from
> other machines are very welcome.

## Install

The tool needs Python 3.11 or newer; the interpreters it inspects can be Python 3.8 or newer.

```bash
uv tool install gpu-stack-inspector
pipx install gpu-stack-inspector          # or with pipx
```

## Use

```bash
gpu-stack-inspector compare --python ~/envs/works/bin/python --python ~/envs/broken/bin/python
gpu-stack-inspector inspect --python ~/envs/broken/bin/python      # one environment in detail
gpu-stack-inspector probe --runtime --python ~/envs/broken/bin/python   # opt-in: really runs a tiny CUDA operation
gpu-stack-inspector report --python ~/envs/broken/bin/python --output gpu-report.md   # redacted, for an issue
gpu-stack-inspector rules list                                     # every check; `rules explain CODE` for one
```

Without `--python` it inspects the `python` your shell runs. See the [command reference](https://github.com/BhuwanJoshi-01/gpu-stack-inspector/blob/main/docs/cli.md).

## What it checks

- **The machine**: driver tools, whether the driver answers and which CUDA versions it supports, the GPUs and their
  compute capability, `CUDA_VISIBLE_DEVICES`.
- **Each interpreter**: whether it starts, whether it is the one your shell runs, whether torch is installed where it
  is needed, and which build it is (CUDA with its version, CPU-only, ROCm), decided from the version tag,
  `torch/version.py`, the CUDA libraries in `torch/lib`, the `nvidia-*` requirements and conda's records, without
  importing torch.
- **Compatibility**: the driver against the CUDA version torch was built for and against nvcc, using dated,
  source-tagged NVIDIA compatibility data.
- **The comparison**: every field that differs, and whether the environments have different PyTorch builds.
- **At runtime** (`probe --runtime` only): importing torch and running an operation on the GPU, step by step.

Every finding has a stable code, the evidence it rests on, what to do next and the command to check it afterwards.
[Troubleshooting](https://github.com/BhuwanJoshi-01/gpu-stack-inspector/blob/main/docs/troubleshooting.md) explains the common ones.

## What it promises

- **Read-only.** It never installs, upgrades or changes anything, and never needs `sudo`. It writes nothing except a
  report file you name.
- **Your environments stay untouched.** A fixed probe program reads each interpreter's metadata. torch is only
  imported with `probe --runtime`, because creating a CUDA context uses GPU memory and can hang on a broken system.
- **Unknown is not a pass.** A check that could not run is unknown or not tested, never green.
- **Shareable output.** Home paths, user and host names, secrets and tokens are redacted before anything is printed
  or written, JSON included, and each report says what was redacted. No network access, no telemetry.

Details in [privacy](https://github.com/BhuwanJoshi-01/gpu-stack-inspector/blob/main/docs/privacy.md).

## Tested on

An Ubuntu 26.04 laptop with an RTX 3050 (driver 595.99.02) and five real environments: conda and uv, Python 3.8 to
3.13, CUDA 11.3 to 13.0 builds and a CPU-only build. CI runs the tests on Linux and Windows and the probe in Python
3.8 to 3.10 and PyPy. Windows and WSL with a GPU, containers and macOS are **not validated**; see the
[support matrix](https://github.com/BhuwanJoshi-01/gpu-stack-inspector/blob/main/docs/support-matrix.md).

## Alternatives

If you only need to check a single environment, these may already be enough:

- [cuda-doctor](https://pypi.org/project/cuda-doctor/) checks the GPU, driver, toolkit, compiler and PyTorch of the
  environment it is installed in.
- [env-doctor](https://github.com/mitulgarg/env-doctor) checks the driver, CUDA, cuDNN and frameworks, and compares
  machines in a fleet dashboard.
- [cudadoc](https://www.cudadoc.com/docs/quickstart) includes a GPU validation suite.

GPU Stack Inspector focuses on comparing **environments on the same machine**.

## Documentation

[Command reference](https://github.com/BhuwanJoshi-01/gpu-stack-inspector/blob/main/docs/cli.md) · [Troubleshooting](https://github.com/BhuwanJoshi-01/gpu-stack-inspector/blob/main/docs/troubleshooting.md) · [Privacy](https://github.com/BhuwanJoshi-01/gpu-stack-inspector/blob/main/docs/privacy.md) ·
[Support matrix](https://github.com/BhuwanJoshi-01/gpu-stack-inspector/blob/main/docs/support-matrix.md) · [JSON output](https://github.com/BhuwanJoshi-01/gpu-stack-inspector/blob/main/docs/json-output.md) · [Architecture](https://github.com/BhuwanJoshi-01/gpu-stack-inspector/blob/main/docs/architecture.md) ·
[Sample reports](https://github.com/BhuwanJoshi-01/gpu-stack-inspector/tree/main/docs/sample-reports) · [Roadmap](https://github.com/BhuwanJoshi-01/gpu-stack-inspector/blob/main/docs/roadmap.md)

## Contributing

Test reports, fixtures from unusual systems, bug reports and pull requests are welcome; see
[CONTRIBUTING.md](https://github.com/BhuwanJoshi-01/gpu-stack-inspector/blob/main/CONTRIBUTING.md). Security problems, including a report that leaks private data, go through
[SECURITY.md](https://github.com/BhuwanJoshi-01/gpu-stack-inspector/blob/main/SECURITY.md).

## License

[MIT](https://github.com/BhuwanJoshi-01/gpu-stack-inspector/blob/main/LICENSE)
