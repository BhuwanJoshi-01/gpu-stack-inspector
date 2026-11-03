# Contributing

Thanks for helping! By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). The most useful
contributions right now:

- **Test reports** from real machines: your GPU, driver, operating system and two Python environments that behave
  differently. Open a *Test report* issue, even when everything works.
- **Fixtures** for situations the tests don't cover yet (unusual `nvidia-smi` output, odd wheel metadata, broken
  interpreters). Sanitize them first: no user names, home paths or tokens.
- **Bug reports and pull requests.**

## Development setup

You need [uv](https://docs.astral.sh/uv/). No GPU is needed for development; the unit tests run on fixtures.

```bash
git clone https://github.com/BhuwanJoshi-01/gpu-stack-inspector.git
cd gpu-stack-inspector
uv sync
uv run gpu-stack-inspector --help
```

Before every commit, run the same checks as CI:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q
```

## How the code is organised

The tool is split into layers that only talk in one direction:

1. **CLI** parses arguments and wires the other layers together. No diagnosis logic lives here.
2. **Collectors** read facts (command output, files, a probe program run inside a target interpreter). They never
   decide what a fact means, and they return a recorded failure instead of raising when a probe goes wrong.
3. **Snapshot**: an immutable, typed record of the facts, including the ones that are unknown.
4. **Rules** are pure functions from a snapshot to findings. They never run commands or read files, so they can be
   tested without hardware.
5. **Renderers** turn findings into terminal, JSON or Markdown output, after redaction.

## Rules for changes

- **Read-only.** Nothing the tool does may change the system: no installs, no driver changes, no writes except the
  report file the user asked for, no `sudo`.
- **Subprocesses** take an argument list (never `shell=True`), a timeout and an output limit, and go through the
  command runner so tests can replace them.
- **Never import a target environment's packages** into the tool's own interpreter. Probes run inside the target
  interpreter as a subprocess. Importing PyTorch is allowed only in the opt-in runtime probe.
- **Unknown is not pass.** If a probe fails or a fact is missing, the result is `unknown`, never a green check.
- **Redact before serializing.** Every output format goes through the redactor. Never add a field that can carry a
  raw environment dump, a user name or a secret.
- **Tests first** where you can: add the tests marked `@pytest.mark.xfail(strict=True, reason="GPU-0xx")`, then
  implement the behaviour and remove the marker. Include negative cases (missing tools, timeouts, garbage output).

## Commits and pull requests

- One logical change per commit, with an imperative subject (at most 72 characters) and a body that explains why.
- Mention the requirement ID when there is one, as a `Refs: GPU-003` trailer.
- The pull request template asks whether you tested on a real machine; say so honestly, "no" is fine.
