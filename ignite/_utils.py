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
        raise TypeError(("input must contain tensors, numbers, dicts or lists; found {}"
                         .format(type(input_))))


def to_onehot(indices, num_classes):
    onehot = torch.zeros(indices.size(0), num_classes, device=indices.device)
    return onehot.scatter_(1, indices.unsqueeze(1), 1)


class RewindableBatchSampler(object):
    """Deterministic rewindable batch sampler

    Args:
        batch_sampler: batch sampler same as used with torch.utils.data.DataLoader
        start_batch_index (int): batch index to start from
    """
    def __init__(self, batch_sampler, start_batch_index=0):
        assert 0 <= start_batch_index < len(batch_sampler)
        self.batch_sampler = batch_sampler
        self.start_batch_index = start_batch_index
        self.batch_indices = self._setup_batch_indices(batch_sampler, start_batch_index)

    @staticmethod
    def _setup_batch_indices(batch_sampler, start_batch_index):
        batch_indices = []
        for batch in batch_sampler:
            batch_indices.append(batch)
        return batch_indices[start_batch_index:]

    def __iter__(self):
        for batch in self.batch_indices:
            yield batch

    def __len__(self):
        return len(self.batch_sampler)
