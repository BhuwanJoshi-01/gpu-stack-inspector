# Security Policy

## Supported versions

Only the latest release and the `main` branch receive fixes.

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them privately through GitHub instead:
**Security → Report a vulnerability** on this repository
(<https://github.com/BhuwanJoshi-01/gpu-stack-inspector/security/advisories/new>).

Include what you found, how to reproduce it, and the impact you expect. You will get an answer within a week,
and a fix or mitigation as soon as possible after that.

## What counts as a security problem here

GPU Stack Inspector runs programs and reads files on your machine, and its reports are meant to be shared in
public issue trackers. These are in scope:

- **Leaks in reports.** A JSON, Markdown or terminal report that contains a secret (token, password, API key),
  your user name, your home directory or a full environment dump. Reports must be safe to paste by default.
- **Code execution beyond what you asked for.** The tool runs a fixed probe program inside each Python
  interpreter you name, and imports PyTorch only with `probe --runtime`. Anything else it executes, imports or
  loads (for example a module from the current directory) is a bug.
- **Changes to your system.** Inspection is read-only. Any write outside the report file you asked for, any
  package installation, driver change or privilege escalation is a bug.
- **Network access.** Normal use makes no network connections.

Bugs in PyTorch, CUDA or the NVIDIA driver themselves should be reported to those projects.
