# nvidia-smi fixtures

| File | Origin | Command |
| --- | --- | --- |
| `rtx3050-laptop-595.query.csv` | real: RTX 3050 Laptop, driver 595.99.02, Ubuntu 26.04 | `nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits` |
| `rtx3050-laptop-595.version.txt` | real, same machine | `nvidia-smi --version` |
| `rtx3050-laptop-595.header.txt` | real, same machine (first 3 lines) | `nvidia-smi` |
| `invalid-field.stdout.txt` | real, same machine; exit code 2, printed on **stdout** | `nvidia-smi --query-gpu=index,name,bogus_field …` |
| `driver-not-loaded.stdout.txt` | synthetic: nvidia-smi's message when the kernel module is not loaded (exit code 9) | `nvidia-smi …` |
| `no-devices.stdout.txt` | synthetic: driver loaded, no GPU visible (exit code 6) | `nvidia-smi …` |
| `two-gpus.query.csv` | synthetic: two different GPUs in one machine | as the query above |
| `not-available.query.csv` | synthetic: an older GPU/driver answering `[N/A]` | as the query above |

Real captures are taken with `LC_ALL=C`. They contain no GPU UUIDs or serial numbers: the tool does not query them.
