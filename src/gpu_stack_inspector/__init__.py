"""GPU Stack Inspector: explain GPU differences between Python environments."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gpu-stack-inspector")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0+unknown"
