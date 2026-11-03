from gpu_stack_inspector.collectors.environment import ALLOWLIST, collect_cuda_environment
from gpu_stack_inspector.models import EnvironmentVariable


def test_only_allowlisted_variables_are_collected() -> None:
    environ = {
        "HF_TOKEN": "hf_secret",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "HOME": "/home/alice",
        "CUDA_HOME": "/usr/local/cuda",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    assert collect_cuda_environment(environ) == (
        EnvironmentVariable("CUDA_VISIBLE_DEVICES", "0"),
        EnvironmentVariable("CUDA_HOME", "/usr/local/cuda"),
    )


def test_an_empty_value_is_kept_because_it_hides_every_gpu() -> None:
    assert collect_cuda_environment({"CUDA_VISIBLE_DEVICES": ""}) == (
        EnvironmentVariable("CUDA_VISIBLE_DEVICES", ""),
    )


def test_unset_variables_are_omitted() -> None:
    assert collect_cuda_environment({}) == ()


def test_the_allowlist_has_no_secret_looking_names() -> None:
    for name in ALLOWLIST:
        assert not any(word in name for word in ("TOKEN", "KEY", "SECRET", "PASS"))
