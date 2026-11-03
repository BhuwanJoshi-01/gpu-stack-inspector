# Command reference

```text
gpu-stack-inspector inspect [--python PATH]… [--format text|json]
gpu-stack-inspector compare --python PATH --python PATH […] [--all] [--format text|json]
gpu-stack-inspector probe --runtime [--python PATH]… [--format text|json]
gpu-stack-inspector report [--python PATH]… [--format markdown|json] [--output FILE] [--force]
gpu-stack-inspector rules list
gpu-stack-inspector rules explain CODE
gpu-stack-inspector --version
```

Without `--python`, commands use the `python` on your `PATH` (what your shell runs), not the tool's own interpreter.
`--python` accepts absolute or relative paths and `~`; symlinks are kept, so a venv's `bin/python` stays in its venv.
On Windows `Scripts\python` also finds `python.exe`.

| Command | What it does |
| --- | --- |
| `inspect` | The machine (OS, driver, GPUs, toolkits, CUDA variables) and each interpreter, then every check. Metadata only. |
| `compare` | `inspect` for two or more interpreters plus a table of the fields that differ (`--all`: every field) and GSI-DIFF-001. |
| `probe --runtime` | Additionally imports torch and runs a tiny CUDA operation in each interpreter with torch (GSI-TORCH-003). Nothing runs without `--runtime`. |
| `report` | A redacted report for a bug tracker: Markdown (default) or JSON. Printed unless `--output FILE`; an existing file is only overwritten with `--force`. Includes the comparison with two or more interpreters. |
| `rules list` / `rules explain CODE` | Every check, and the full explanation and sources of one. |

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | The inspection ran. Findings, even failures, do not change this. |
| 1 | Reserved for `--strict` (not available yet). |
| 2 | Usage error (unknown option, fewer than two interpreters for `compare`, `probe` without `--runtime`, an existing report file without `--force`) or an internal error. |
| 3 | A named interpreter could not be started (missing, not executable). The report is still printed. |
| 130 | Interrupted with Ctrl+C. |

Internal errors print one redacted line; set `GSI_DEBUG=1` to see the traceback for a bug report.

JSON output is described in [json-output.md](json-output.md).
