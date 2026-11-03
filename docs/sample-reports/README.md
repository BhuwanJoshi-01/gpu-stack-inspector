# Sample reports

Real `report` output from the RTX 3050 laptop in the [support matrix](../support-matrix.md), comparing a conda
environment with torch 2.11.0+cu128 against a uv venv with torch 2.15.0+cpu:

```bash
gpu-stack-inspector report --python ~/envs/sapiens2/bin/python --python ~/envs/cpu-venv/bin/python --output compare-cpu-only.md
gpu-stack-inspector report --python … --python … --format json --output compare-cpu-only.json
```

Paths were shortened to `~/envs/…` after generation; everything else is exactly what the tool wrote.

- [compare-cpu-only.md](compare-cpu-only.md): the Markdown dossier, as it would be pasted into an issue.
- [compare-cpu-only.json](compare-cpu-only.json): the same run as JSON (valid against `docs/json-schema-v1.json`).
