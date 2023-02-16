import torch


def denormalize(t, mean, std, max_pixel_value=255):
    assert isinstance(t, torch.Tensor), f"{type(t)}"
    assert t.ndim == 3
    d = t.device
    mean = torch.tensor(mean, device=d).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std, device=d).unsqueeze(-1).unsqueeze(-1)
    tensor = std * t + mean
    tensor *= max_pixel_value
    return tensor
