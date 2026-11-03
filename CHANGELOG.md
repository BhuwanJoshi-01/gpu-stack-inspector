# Changelog

All notable changes to this project are listed here. Versions follow
[Semantic Versioning](https://semver.org/); the JSON report format has its own
`schema_version` and changes only with a new major version of the schema.

## Unreleased

Nothing released yet. The first release (0.1.0) will inspect and compare
explicitly given Python interpreters.

- `gpu-stack-inspector --help` and `--version`. Exit codes: 0 when the
  inspection ran (whatever it found), 1 for `--strict` failures, 2 for usage
  errors and internal errors, 3 when required context is missing, 130 for
  Ctrl+C. Internal errors print one redacted line; `GSI_DEBUG=1` shows the
  traceback.
