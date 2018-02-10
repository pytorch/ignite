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
