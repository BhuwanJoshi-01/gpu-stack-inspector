"""Fake torch: a build without kernels for this GPU (wrong compute capability)."""

__version__ = "2.11.0+cu128"


class version:  # lower case, like torch.version
    cuda = "12.8"


class _Tensor:
    def __init__(self, values):
        self.values = values

    def __mul__(self, factor):
        return _Tensor([v * factor for v in self.values])

    def sum(self):
        return _Tensor([sum(self.values)])

    def cpu(self):
        return self

    def item(self):
        return self.values[0]


def ones(count, device=None):
    raise RuntimeError(
        "CUDA error: no kernel image is available for execution on the device\n"
        "CUDA kernel errors might be asynchronously reported at some other API call"
    )


class cuda:  # lower case, like torch.cuda
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def get_device_name(index):
        return "NVIDIA GeForce RTX 3050 Laptop GPU"

    @staticmethod
    def get_device_capability(index):
        return (8, 6)

    @staticmethod
    def synchronize():
        return None
