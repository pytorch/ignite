import collections

import torch
from torch._six import string_classes


def _to_hours_mins_secs(time_taken):
    mins, secs = divmod(time_taken, 60)
    hours, mins = divmod(mins, 60)
    return hours, mins, secs


def convert_tensor(input_, device=None):
    if torch.is_tensor(input_):
        if device:
            input_ = input_.to(device=device)
        return input_
    elif isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: convert_tensor(sample, device=device) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [convert_tensor(sample, device=device) for sample in input_]
    else:
        raise TypeError(("input must contain tensors, dicts or lists; found {}"
                         .format(type(input_))))


def to_onehot(indices, num_classes):
    onehot = torch.zeros(indices.size(0), num_classes, device=indices.device)
    return onehot.scatter_(1, indices.unsqueeze(1), 1)
