# Runtime probe captures

Real output of `probes/runtime_program.py` on one laptop (RTX 3050 Laptop GPU, compute capability 8.6, driver
595.99.02 supporting CUDA up to 13.2, Ubuntu 26.04). The program prints no paths, so nothing needed redacting.

| File | Environment | Result |
| --- | --- | --- |
| `sapiens2.stdout.txt` | conda, Python 3.13, torch 2.11.0+cu128 | every step passes |
| `wilor.stdout.txt` | conda, Python 3.10, torch from PyPI | every step passes; at runtime `torch.__version__` is `2.14.1+cu130` although the package metadata says `2.14.1` |
| `smplerx.stdout.txt` | conda, Python 3.8, conda's torch 1.12.0 (CUDA 11.3) | every step passes |
| `cpu-venv.stdout.txt` | uv venv, Python 3.12, torch 2.15.0+cpu | stops at "cuda available": False |
