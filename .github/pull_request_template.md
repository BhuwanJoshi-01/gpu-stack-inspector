## What does this change?

<!-- What and why, from the user's point of view. Link the issue if there is one. -->

Requirement IDs: <!-- e.g. GPU-003, or "none" -->

## How was it tested?

<!-- Unit tests and fixtures added. -->

Tested on a real machine: <!-- yes (GPU, driver, OS) / no -->

## Impact

- Privacy: <!-- Can any new field carry a path, user name, secret or env value? How is it redacted? -->
- Compatibility: <!-- Changes to CLI flags, exit codes or the JSON schema? -->
- Follow-up issue: <!-- if anything is left for later -->

## Checklist

- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q` passes
- [ ] Nothing new changes the system or reaches the network; subprocesses go through the command runner
- [ ] Negative cases are tested (missing tool, timeout, garbage output, permission denied)
- [ ] CHANGELOG.md and docs updated if behaviour, flags or output changed
