# AGENTS.md

Guide for contributors and coding agents. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the rules in full.

## What this is

A read-only Python CLI that inspects several Python interpreters and explains why PyTorch can use the GPU in one
and not another. Plain Python 3.11+, Typer and Rich, packaged with uv (`uv_build` backend), `src/` layout.

## Commands

```bash
uv sync                                    # create .venv with the dev tools
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q   # what CI runs
uv run pytest -q --cov                     # with coverage
GSI_INTEGRATION=1 uv run pytest -q tests/integration   # against this machine's GPU/driver
GSI_TEST_PYTHONS="$(uv python find --no-project 3.9):$HOME/env/bin/python" uv run pytest -q tests/integration
                                                       # probe inside other interpreters (CI: 3.8-3.10, PyPy)
HYPOTHESIS_PROFILE=thorough uv run pytest -q           # property tests with 2000 examples
```

## Layout

```
src/gpu_stack_inspector/
  cli.py          Typer app + main(): exit codes, redacted internal errors (GSI_DEBUG=1 for tracebacks)
  models.py       Status/Severity/Confidence, Evidence, Finding (ids GSI-<AREA>-<NNN>), ProbeFailure, Snapshot
  runner.py       CommandRunner protocol, SubprocessRunner (no shell, timeout kills the process group,
                  capped output), probe_failure()
  privacy.py      strip_control(), RedactionContext, Redactor (text + whole snapshots, summary)
  exit_codes.py   ExitCode: 0 ok, 1 strict, 2 error, 3 context unavailable, 130 interrupted
  orchestrator.py inspect_host() / inspect(): runs the collectors and the interpreter probe,
                  assembles the Snapshot (no diagnosis); default_interpreters()
  probes/         metadata_program.py: runs INSIDE target interpreters (Python 3.8+, stdlib only,
                  never imports target packages, prints one JSON doc between markers);
                  metadata_probe.py: runs it in an empty temp cwd and parses the answer defensively
  reports/        json_report.py, terminal.py (plain Text, never Rich markup from collected data)
  collectors/     host.py (OS, distro, WSL flag), nvidia.py (nvidia-smi query, --version, table
                  header fallback), toolkit.py (nvcc, /usr/local/cuda*, /opt/cuda*, CUDA_HOME…),
                  environment.py (allowlisted CUDA variables only), python_env.py (--python paths;
                  symlinks are never resolved, a venv's python must stay in its venv)
tests/
  conftest.py     autouse no_network guard; HYPOTHESIS_PROFILE=thorough
  fakes.py        FakeRunner: scripted commands only, anything else fails the test
  fixtures/       real and synthetic captures, each set with a README saying which is which
  unit/           unit tests (no GPU needed)
  integration/    real-machine tests, only with GSI_INTEGRATION=1
docs/roadmap.md   requirement IDs GPU-001 … and what v0.1 includes
.github/          CI, issue forms, PR template
```

## Rules that are easy to break

- **The probe program is not the tool.** `probes/*_program.py` must stay Python 3.8 syntax and stdlib
  only (tests parse it with `feature_version=(3, 8)` and check its imports; CI runs it in 3.8-3.10 and PyPy).

- **Never import a target environment's packages** (torch, nvidia-*) into this process. Probes run as subprocesses
  of the target interpreter. Importing torch is only allowed in the opt-in runtime probe.
- **Read-only.** No installs, no writes except a report file the user named, no `sudo`, no network access.
- Subprocesses: argument lists only (no `shell=True`), always a timeout and an output cap, always through the
  command runner so tests can fake them.
- Rules are pure functions of a snapshot; they never run commands or read files.
- A failed or missing probe yields `unknown`, never `pass`.
- Redact before serializing; every renderer, JSON included, gets redacted data.
- Tests first where possible: `@pytest.mark.xfail(strict=True, reason="GPU-0xx")`, then implement and drop the
  marker. Cover negative cases (missing tool, timeout, garbage output, permission denied).

## Commits

One logical change per commit; imperative subject of at most 72 characters; a body that explains why; a
`Refs: GPU-0xx` trailer when a requirement applies. Every commit passes the CI command above.
