import torch
from packaging.version import Version

from ignite.distributed.comp_models.base import _torch_version_gt_112


_torch_version_lt_2 = Version(torch.__version__) < Version("2")


def is_mps_available_and_functional():
    if not _torch_version_gt_112:
        return False

    if not torch.backends.mps.is_available():
        return False
    try:
        # Try to allocate a small tensor on the MPS device
        torch.tensor([1.0], device="mps")
        return True
    except RuntimeError:
        return False
