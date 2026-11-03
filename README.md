# GPU Stack Inspector

**Explain why your AI code uses the GPU in one Python environment but not another.**

> **Status: in development.** Nothing is released yet. The commands below are the planned interface for 0.1.0;
> see the [roadmap](docs/roadmap.md).

`nvidia-smi` shows your GPU, yet `torch.cuda.is_available()` is `False` in one environment and `True` in the
other. Usually the cause is a CPU-only wheel, the wrong interpreter, or conflicting CUDA libraries. Tools that
check one environment at a time make you run them in each one and compare the output by hand.

GPU Stack Inspector inspects **several interpreters from one command** and shows the differences side by side,
with the evidence for each finding:

```bash
gpu-stack-inspector inspect --python ~/envs/works/bin/python --python ~/envs/broken/bin/python   # works today
gpu-stack-inspector compare --python ~/envs/works/bin/python --python ~/envs/broken/bin/python
gpu-stack-inspector probe --runtime --python ~/envs/broken/bin/python   # opt-in: really runs a tiny CUDA op
gpu-stack-inspector report --format markdown --output gpu-report.md     # redacted, safe to paste in an issue
```

## What it promises

- **Read-only.** It reads files and runs bounded diagnostic commands. It never installs, upgrades or changes
  anything, and never needs `sudo`.
- **Your environments stay untouched.** A fixed probe program runs inside each interpreter you name. PyTorch is
  only imported when you ask for `probe --runtime`, because creating a CUDA context can use GPU memory or hang on
  a broken system.
- **Metadata and runtime are reported separately.** "This wheel is a CUDA build" and "a CUDA operation actually
  ran" are different findings, and the report says which one was tested.
- **Unknown is not a pass.** A check that could not run is shown as unknown, never as green.
- **Shareable reports.** User names, home paths and secret-looking values are redacted before any output is
  written, JSON included. No network access, no telemetry.

## Alternatives

If you only need to check a single environment, these may already be enough:

- [cuda-doctor](https://pypi.org/project/cuda-doctor/) checks the GPU, driver, toolkit, compiler and PyTorch of the
  environment it is installed in.
- [env-doctor](https://github.com/mitulgarg/env-doctor) checks the driver, CUDA, cuDNN and frameworks, and compares
  machines in a fleet dashboard.
- [cudadoc](https://www.cudadoc.com/docs/quickstart) includes a GPU validation suite.

GPU Stack Inspector focuses on comparing **environments on the same machine**.

## License

[MIT](LICENSE)
