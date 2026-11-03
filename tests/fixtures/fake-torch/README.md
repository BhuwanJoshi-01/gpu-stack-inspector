# Fake torch packages

Minimal stand-ins for PyTorch, put on `PYTHONPATH` to test the runtime probe without a GPU or a real
torch install. Each implements only what `probes/runtime_program.py` calls.

| Folder | Behaves like |
| --- | --- |
| `working/` | a CUDA build on a working GPU: the tensor operation succeeds |
| `cpu-only/` | a CPU-only build: `torch.cuda.is_available()` is False |
| `no-kernel-image/` | a build without kernels for this GPU: the first CUDA operation raises PyTorch's real error message |
| `import-error/` | a damaged install: `import torch` fails on a missing CUDA library |
