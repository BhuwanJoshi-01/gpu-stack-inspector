"""Fake torch: a CPU-only build."""

__version__ = "2.15.0+cpu"


class version:  # lower case, like torch.version
    cuda = None


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
    return _Tensor([1.0] * count)


class cuda:  # lower case, like torch.cuda
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0

    @staticmethod
    def get_device_name(index):
        return "NVIDIA GeForce RTX 3050 Laptop GPU"

    @staticmethod
    def get_device_capability(index):
        return (8, 6)

    @staticmethod
    def synchronize():
        return None
