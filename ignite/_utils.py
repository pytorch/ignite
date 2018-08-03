import collections

import torch
from torch._six import string_classes


def _to_hours_mins_secs(time_taken):
    """Convert seconds to hours, mins, and seconds."""
    mins, secs = divmod(time_taken, 60)
    hours, mins = divmod(mins, 60)
    return hours, mins, secs


def convert_tensor(input_, device=None):
    """Move tensors to relevant device."""
    def _func(tensor):
        return tensor.to(device=device) if device else tensor

    return apply_func_tensor(input_, _func)


def apply_func_tensor(input_, func):
    """Apply a funcction of a tensor or mapping, or sequence of tensors."""
    if torch.is_tensor(input_):
        return func(input_)
    elif isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: apply_func_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [apply_func_tensor(sample, func) for sample in input_]
    else:
        raise TypeError(("input must contain tensors, dicts or lists; found {}"
                         .format(type(input_))))


def to_onehot(indices, num_classes):
    """Convert a tensor of indices to a tensor of one-hot indicators."""
    onehot = torch.zeros(indices.size(0), num_classes, device=indices.device)
    return onehot.scatter_(1, indices.unsqueeze(1), 1)
