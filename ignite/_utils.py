import collections

import torch
from torch._six import string_classes
from torch.autograd import Variable


def _to_hours_mins_secs(time_taken):
    mins, secs = divmod(time_taken, 60)
    hours, mins = divmod(mins, 60)
    return hours, mins, secs


def to_variable(batch, cuda=False, volatile=False):
    if torch.is_tensor(batch):
        if cuda:
            batch = batch.cuda()
        return Variable(batch, volatile=volatile)
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, collections.Mapping):
        return {k: to_variable(sample, cuda=cuda, volatile=volatile) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [to_variable(sample, cuda=cuda, volatile=volatile) for sample in batch]
    else:
        raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                         .format(type(batch[0]))))


def to_tensor(variable, cpu=False):
    if isinstance(variable, Variable):
        t = variable.data
        if cpu:
            t = t.cpu()
        return t
    elif isinstance(variable, string_classes):
        return variable
    elif isinstance(variable, collections.Mapping):
        return {k: to_tensor(sample, cpu=cpu) for k, sample in variable.items()}
    elif isinstance(variable, collections.Sequence):
        return [to_tensor(sample, cpu=cpu) for sample in variable]
    else:
        raise TypeError(("Argument variable must contain torch.autograd.Variable, numbers, dicts or lists; found {}"
                         .format(type(variable))))


def to_onehot(indices, num_classes):
    onehot = torch.zeros(indices.size(0), num_classes)
    if indices.is_cuda:
        onehot = onehot.cuda()
    return onehot.scatter_(1, indices.unsqueeze(1), 1)
