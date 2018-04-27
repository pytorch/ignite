import collections

import torch
from torch._six import string_classes
from torch.autograd import Variable


def _to_hours_mins_secs(time_taken):
    mins, secs = divmod(time_taken, 60)
    hours, mins = divmod(mins, 60)
    return hours, mins, secs


def to_tensor(variable, cuda=False, requires_grad=False):
    if torch.is_tensor(variable):
        if cuda:
            variable = variable.cuda()
        if not requires_grad:
            variable = variable.detach()
        return variable
    elif isinstance(variable, string_classes):
        return variable
    elif isinstance(variable, collections.Mapping):
        return {k: to_tensor(sample, cuda=cuda, requires_grad=requires_grad) for k, sample in variable.items()}
    elif isinstance(variable, collections.Sequence):
        return [to_tensor(sample, cuda=cuda, requires_grad=requires_grad) for sample in variable]
    else:
        raise TypeError(("Argument variable must contain torch.autograd.Variable, numbers, dicts or lists; found {}"
                         .format(type(variable))))


def to_onehot(indices, num_classes):
    onehot = torch.zeros(indices.size(0), num_classes)
    if indices.is_cuda:
        onehot = onehot.cuda()
    return onehot.scatter_(1, indices.unsqueeze(1), 1)
