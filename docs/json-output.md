# JSON output

`inspect`, `compare` and `report` print a JSON document with `--format json`. Its layout is described by
[`json-schema-v1.json`](json-schema-v1.json) (JSON Schema draft 2020-12) and checked against it in the test suite.

## Stability

- `schema_version` is `1`. Within a schema version, fields are not renamed, removed or given a different type, and no
  field is added without updating the schema: every object in the schema forbids unknown properties, and a test
  compares the schema with the code.
- An incompatible change means a new schema version (`json-schema-v2.json`) and a note in the changelog.
- Finding codes (`GSI-TORCH-002` …) are stable. A check that is replaced gets a new code; old codes are not reused.
- `tool_version` is the version of the tool that wrote the document; it does not affect the layout.

## What is in it

| Key | Contents |
| --- | --- |
| `host` | operating system, architecture, kernel, distribution, WSL flag |
| `nvidia` | path of nvidia-smi, driver version, the newest CUDA version the driver supports (`max_cuda_version`, not an installed toolkit), GPUs, failures |
| `toolkit` | nvcc and its version, CUDA toolkit directories |
| `cuda_environment` | allowlisted CUDA-related environment variables that are set |
| `interpreters` | per interpreter: version, prefixes, conda/venv, GPU-relevant packages, torch evidence, the derived `torch_build` verdict and `virtualenv` flag, failures |
| `comparison` | field-by-field rows (`compare` only), each with a `differs` flag |
| `findings` | every check's result: `id`, `status`, `severity`, `confidence`, `summary`, `reason`, `subject` (`host`, an interpreter label, or `comparison`), `evidence`, `recommended_action`, `verify_command`, `limitations`, `caused_by` |
| `probe_failures` | everything that could not be collected, with the reason |
| `redactions` | how many values of each kind were redacted before output |

Unknown facts are `null`, never a guess. Status values: `pass`, `warn`, `fail`, `unknown`, `not_applicable`,
`not_tested`; only `pass` means a check ran and succeeded.

## Privacy

The document is redacted before it is written: home directories become `~`, user and host names are removed, and
secret values, URL credentials and tokens are replaced. `redactions` says what was removed. Check the document
before sharing it anyway; see [privacy](../SECURITY.md).
