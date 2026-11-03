# Support matrix

Only what was actually run is listed as tested. Everything else may work, but is not claimed. Test reports from other
systems are very welcome (open a *Test report* issue).

## Tested on real hardware

| Host | GPU and driver | Environments | Result |
| --- | --- | --- | --- |
| Ubuntu 26.04.1 LTS, x86_64, kernel 7.0 | RTX 3050 Laptop GPU (compute 8.6), driver 595.99.02 (CUDA up to 13.2), Ubuntu's nvcc 12.4 | conda: torch 2.11.0+cu128 (Python 3.13), PyPI torch 2.14.1 / CUDA 13.0 (Python 3.10), conda's torch 1.12.0 / CUDA 11.3 (**Python 3.8**), envs without torch; uv venv: torch 2.15.0+cpu (Python 3.12) | every check and `probe --runtime` behave as documented; the CPU-only build is caught statically and at runtime |

## Tested in CI (no GPU)

| Platform | What runs |
| --- | --- |
| Ubuntu, Python 3.11–3.14 | the full test suite, fixtures from the machine above |
| Windows, Python 3.13 | the test suite, and the built wheel installed into a fresh environment and run |
| Target interpreters CPython 3.8, 3.9, 3.10 and PyPy 3.10 | the metadata probe answers correctly |

## Not validated

- **Windows with an NVIDIA GPU**, **WSL2**, **containers** (NVIDIA Container Toolkit): the tool runs, but its GPU
  findings there are not validated. WSL is detected and reported as a fact only.
- **macOS**: no NVIDIA CUDA; the PyTorch checks still run, the driver checks do not apply.
- **ROCm builds**: recognised and named, not judged.
- **pyenv, Poetry, pixi environments**: should work like any interpreter path; not tested yet.
- **Multi-GPU machines**: parsed from synthetic `nvidia-smi` output only.

## Requirements

The tool needs Python 3.11 or newer for itself. The interpreters it inspects can be Python 3.8 or newer (older ones
are reported as too old for the probe).
