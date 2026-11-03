"""Whatever the evidence, the classifier gives a coherent verdict and never raises."""

from __future__ import annotations

import re

from hypothesis import given
from hypothesis import strategies as st

from gpu_stack_inspector.models import Confidence, TorchFacts
from gpu_stack_inspector.rules.torch_build import TorchBuildKind, classify_torch

versions = st.one_of(
    st.text(max_size=20),
    st.sampled_from(
        ["2.11.0+cu128", "2.11.0+cpu", "2.9.1+rocm6.4", "2.14.1", "1.7.1+cu92", "2.0+cu"]
    ),
)
records = st.lists(
    st.one_of(
        st.text(max_size=30),
        st.sampled_from(
            [
                "pytorch-1.12.0-py3.8_cuda11.3_cudnn8.3.2_0",
                "pytorch-2.3.0-py3.11_cpu_0",
                "pytorch-mutex-1.0-cuda",
                "pytorch-mutex-1.0-cpu",
                "pytorch-cuda-11.8-h7e8668a_6",
                "pytorch-2.4.1-cuda118_py310h6b1e8c4_300",
            ]
        ),
    ),
    max_size=4,
)
facts = st.builds(
    TorchFacts,
    version=versions,
    requires_nvidia=st.lists(
        st.sampled_from(["nvidia-cudnn-cu12", "nvidia-nccl-cu13"]), max_size=2
    ).map(tuple),
    cuda_library=st.one_of(st.none(), st.booleans()),
    hip_library=st.one_of(st.none(), st.booleans()),
    version_py_found=st.booleans(),
    built_cuda=st.one_of(st.none(), st.sampled_from(["11.3", "12.8", "13.0"])),
    built_hip=st.one_of(st.none(), st.just("6.4.43482")),
    conda_records=st.one_of(st.none(), records.map(tuple)),
)


@given(facts)
def test_verdicts_are_coherent(torch: TorchFacts) -> None:
    build = classify_torch(torch)
    assert build.kind is not TorchBuildKind.MISSING
    assert build.evidence
    if build.cuda_version is not None:
        assert build.kind is TorchBuildKind.CUDA
        assert re.fullmatch(r"[0-9]+\.[0-9]+", build.cuda_version)
    if build.kind is TorchBuildKind.UNKNOWN:
        assert build.confidence is Confidence.LOW
    if build.conflicts:
        assert build.confidence is not Confidence.HIGH
