import torch

from ignite.distributed.comp_models.base import _torch_version_gt_112


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
