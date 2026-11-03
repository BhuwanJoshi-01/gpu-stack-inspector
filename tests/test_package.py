from importlib.metadata import version

import gpu_stack_inspector


def test_version_comes_from_installed_metadata() -> None:
    assert gpu_stack_inspector.__version__ == version("gpu-stack-inspector")
