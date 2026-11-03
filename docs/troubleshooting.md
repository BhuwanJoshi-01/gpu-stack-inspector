# Troubleshooting with the findings

Start with `gpu-stack-inspector compare --python <works> --python <broken>`, or `inspect` for one environment. Each
finding says what to do next and how to check it; `gpu-stack-inspector rules explain CODE` explains a check in full.
The most common situations:

| You see | It means | Do |
| --- | --- | --- |
| **FAIL GSI-TORCH-002** "CPU-only build" | This torch has no CUDA support at all; `torch.cuda.is_available()` is False whatever the driver can do. | Install a CUDA build from the selector at pytorch.org, choosing a CUDA version up to what GSI-SYS-002 says the driver supports. |
| **WARN GSI-DIFF-001** "different PyTorch builds" | The environments you compared have different builds; usually the whole answer. | Install the working environment's build in the other one. |
| **FAIL GSI-CUDA-003** "newer than the driver supports" | torch targets a newer CUDA major version than the driver: CUDA will not initialise. | Update the driver to the version named, or install a torch build for the driver's CUDA version. |
| **WARN GSI-CUDA-003** (newer minor version) | Usually works through NVIDIA's minor version compatibility; some calls can fail with `cudaErrorCallRequiresNewerDriver`. | Update the driver if you hit such errors. |
| **WARN GSI-SYS-001** / **FAIL GSI-SYS-002** | No NVIDIA driver tools, or the driver does not answer (module not loaded, reboot pending, Secure Boot). | Fix the driver first; everything else follows from it. |
| **FAIL GSI-SYS-003** "sees no GPU" | The driver works but no GPU is visible: hybrid-graphics mode, BIOS, VM or container passthrough. | Switch the GPU on for this system, then `nvidia-smi -L`. |
| **WARN GSI-ENV-003** | `CUDA_VISIBLE_DEVICES` hides the GPUs (empty, `-1`, or an index that does not exist). | Unset it or use indices from `nvidia-smi -L`. |
| **NOT TESTED GSI-TORCH-004** | Metadata shows a CUDA build, but nothing has run on the GPU yet. | `gpu-stack-inspector probe --runtime --python <path>` |
| **FAIL GSI-TORCH-003** | The runtime check failed at a named step: import, availability, or the tensor operation ("no kernel image" means the build has no code for your GPU's compute capability). | Follow the step-specific advice in the finding. |
| **UNKNOWN GSI-TORCH-002** | The torch install contradicts itself (e.g. a `+cu128` tag but no CUDA library). | Reinstall torch in that environment. |
| **WARN GSI-ENV-002** | You inspected a different python than the one your shell runs. | Activate that environment, or run your program with its full path. |

If a finding is wrong for your system, please open a bug report with the redacted output of
`gpu-stack-inspector report`; false alarms are bugs.
