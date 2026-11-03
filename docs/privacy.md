# Privacy and side effects

Reports from this tool are meant to be pasted into public bug trackers, so it is built to collect little, change
nothing and hide what identifies you.

## What it reads

| Source | What is kept |
| --- | --- |
| `platform`, `/etc/os-release` | OS, architecture, kernel, distribution name |
| `nvidia-smi` | driver version, the newest CUDA it supports, each GPU's index, name, memory and compute capability. Not GPU UUIDs or serials, not the process list |
| `nvcc --version`, CUDA toolkit folders | versions and folder paths |
| Environment variables | only `CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER`, `NVIDIA_VISIBLE_DEVICES`, `NVIDIA_DRIVER_CAPABILITIES`, `CUDA_HOME`, `CUDA_PATH`, `LD_LIBRARY_PATH`, `PYTORCH_CUDA_ALLOC_CONF`. Nothing else is read, so tokens in other variables never reach the tool |
| Each `--python` interpreter | version, prefixes, conda/venv, whether user site and `PYTHONPATH` are in effect, and an allowlist of GPU-relevant packages (torch and companions, JAX, TensorFlow, CuPy, ONNX Runtime, TensorRT, NumPy, `nvidia-*`). Not your other packages |

## What it never does

- Install, upgrade or remove anything; change drivers, files or settings; need `sudo`.
- Connect to the network. The test suite blocks network access to make sure.
- Import packages of the inspected environments, except with `probe --runtime` (below).
- Write files, except the one you name with `report --output`, which it will not overwrite without `--force`, and
  an empty temporary directory per probe that is removed straight away.

## `probe --runtime`

This is the one command that runs code from your environments: it imports torch in each interpreter and runs a
16-element tensor operation on the GPU. That creates a CUDA context and uses a little GPU memory for a moment. It only
runs when you pass `--runtime`; without it the command explains this and does nothing.

## Redaction

Everything is redacted before it is printed or written, JSON included:

- your home directory becomes `~`, other users' home directories `/home/<user>`;
- your user name and host name are removed (short or generic ones such as `root` or `ubuntu` are left alone, as they
  identify nobody);
- values of variables whose names look secret (`*_TOKEN`, `*_KEY`, `PASSWORD`, …), credentials in URLs, and known
  token formats (GitHub, Hugging Face, OpenAI, Anthropic, AWS, Slack, JWT, Bearer) are replaced;
- terminal escape sequences are stripped, so a crafted package name cannot rewrite your terminal.

Each report ends by saying what was redacted. Read it before you share it anyway: redaction catches known patterns,
not everything.
