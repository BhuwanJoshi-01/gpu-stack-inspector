# Metadata probe fixtures

Real output of the metadata probe (`probes/metadata_program.py`), captured on one laptop
(RTX 3050, Ubuntu 26.04, Miniconda) and redacted with the tool's own redactor before
saving: home directories are `~`, no user or host names remain.

| File | Python | What it shows |
| --- | --- | --- |
| `conda-base.stdout.txt` | 3.13.13 | conda base env: accelerate, bitsandbytes, keras, onnxruntime **without torch** |
| `conda-sapiens2.stdout.txt` | 3.13.13 | torch 2.11.0+cu128 (download.pytorch.org wheel, `+cu` tag) with 15 nvidia-* wheels |
| `conda-wilor.stdout.txt` | 3.10.22 | torch 2.14.1 from PyPI: a CUDA build **without** a `+cu` tag, 15 nvidia-* wheels |
| `conda-smplerx.stdout.txt` | 3.8.20 | torch 1.12.0 installed by conda: no `+cu` tag and no nvidia-* wheels |
| `conda-dwpose.stdout.txt` | 3.10.20 | no torch; onnxruntime-gpu with 7 nvidia-* wheels |
