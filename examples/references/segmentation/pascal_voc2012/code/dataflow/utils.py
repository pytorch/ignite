from torch.utils.data import DataLoader, BatchSampler
import torch.utils.data.distributed as data_dist


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
