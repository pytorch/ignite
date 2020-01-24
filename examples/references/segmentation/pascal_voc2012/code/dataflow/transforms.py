import torch

from ignite.utils import convert_tensor


def ignore_mask_boundaries(force_apply, **kwargs):
    assert 'mask' in kwargs, "Input should contain 'mask'"
    mask = kwargs['mask']
    mask[mask == 255] = 0
    kwargs['mask'] = mask
    return kwargs


def denormalize(t, mean, std, max_pixel_value=255):
    assert isinstance(t, torch.Tensor), "{}".format(type(t))
    assert t.ndim == 3
    d = t.device
    mean = torch.tensor(mean, device=d).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std, device=d).unsqueeze(-1).unsqueeze(-1)
    tensor = std * t + mean
    tensor *= max_pixel_value
    return tensor


def prepare_batch_fp32(batch, device, non_blocking):
    x, y = batch['image'], batch['mask']
    x = convert_tensor(x, device, non_blocking=non_blocking)
    y = convert_tensor(y, device, non_blocking=non_blocking).long()
    return x, y
