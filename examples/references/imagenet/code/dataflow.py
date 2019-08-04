import torch
import numpy as np

import torch.utils.data.distributed as data_dist
from torch.utils.data import DataLoader, BatchSampler


def redefine_data_loader(init_data_loader, **kwargs):
    """Method to recreate new dataloader from existing one with modified parameters as batch size, sampler, number of workers etc

    Args:
        init_data_loader (DataLoader): initial torch DataLoader
        **kwargs : kwargs are the same as for torch DataLoader
    """
    assert isinstance(init_data_loader, DataLoader), "{} vs DataLoader".format(type(init_data_loader))
    assert len(kwargs) > 0, "At least a single new option should be provided"

    possible_keys = [
        'dataset', 'batch_size',
        'sampler', 'batch_sampler', 'num_workers',
        'collate_fn', 'pin_memory', 'drop_last',
        'timeout', 'worker_init_fn'
    ]

    assert all([k in possible_keys for k in kwargs.keys()]), "{} vs {}".format(kwargs.keys(), possible_keys)

    new_kwargs = dict(kwargs)
    for k in possible_keys:
        if k not in new_kwargs:
            obj = getattr(init_data_loader, k)
            if k == 'batch_sampler' and \
                    isinstance(obj, BatchSampler) and \
                    obj.__class__ == BatchSampler:
                # Ignore default batch sampler
                continue
            new_kwargs[k] = obj

    # Remove mutually exclusive cases:
    if 'batch_sampler' in new_kwargs:
        for k in ('batch_size', 'sampler', 'drop_last'):
            del new_kwargs[k]

    return DataLoader(**new_kwargs)


def setup_distrib_sampler(data_loader):
    assert isinstance(data_loader.batch_sampler, BatchSampler), \
        "We can not recreate new dataloader with distributed sampler if a non-default batchsampler is setup"

    dist_sampler = data_dist.DistributedSampler(data_loader.dataset)
    return redefine_data_loader(data_loader, sampler=dist_sampler), dist_sampler


# https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


# https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
class DataPrefetcher:

    def __init__(self, loader, mean=None, std=None):

        if mean is not None:
            assert hasattr(mean, "__len__")

        if std is not None:
            assert hasattr(std, "__len__")

        if (mean is not None) and (std is not None):
            assert len(mean) == len(std)

        self.loader = loader
        self.loader_iter = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = None
        self.std = None

        def _to_cuda(v):
            n = len(v)
            return torch.tensor(v).cuda().reshape(1, n, 1, 1)

        if mean is not None:
            self.mean = _to_cuda(mean)

        if std is not None:
            self.std = _to_cuda(std)

        self.next_input = None
        self.next_target = None

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader_iter)
        except StopIteration:
            self.next_input, self.next_target = None, None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True)
            if (self.mean is not None) and (self.std is not None):
                self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is None or target is None:
            raise StopIteration

        input.record_stream(torch.cuda.current_stream())
        target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    next = __next__  # Python 2 compatibility
