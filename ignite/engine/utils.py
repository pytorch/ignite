import inspect

import torch


def _check_signature(engine, fn, fn_description, *args, **kwargs):
    exception_msg = None

    signature = inspect.signature(fn)
    try:
        signature.bind(engine, *args, **kwargs)
    except TypeError as exc:
        fn_params = list(signature.parameters)
        exception_msg = str(exc)

    if exception_msg:
        passed_params = [engine] + list(args) + list(kwargs)
        raise ValueError("Error adding {} '{}': "
                         "takes parameters {} but will be called with {} "
                         "({}).".format(fn, fn_description, fn_params, passed_params, exception_msg))


def _update_dataloader(dataloader, new_batch_sampler):
    params_keys = [k for k in dataloader.__dict__.keys() if not k.startswith("_")]
    for k in ['batch_size', 'sampler', 'drop_last', 'batch_sampler', 'dataset_kind']:
        if k in params_keys:
            params_keys.remove(k)
    params = {k: getattr(dataloader, k) for k in params_keys}
    params['batch_sampler'] = new_batch_sampler
    return torch.utils.data.DataLoader(**params)


class ReproducibleBatchSampler(torch.utils.data.sampler.BatchSampler):
    """Reproducible batch sampler. Internally, this class iterates and stores indices of the input batch sampler.

    Args:
        batch_sampler (torch.utils.data.sampler.BatchSampler): batch sampler same as used with
            `torch.utils.data.DataLoader`
        start_iteration (int, optional): optional start iteration
    """
    def __init__(self, batch_sampler, start_iteration=None):
        if not isinstance(batch_sampler, torch.utils.data.sampler.BatchSampler):
            raise TypeError("Argument batch_sampler should be torch.utils.data.sampler.BatchSampler")

        self.batch_indices = None
        self.batch_sampler = batch_sampler
        self.start_iteration = start_iteration
        self.sampler = self.batch_sampler.sampler

    def setup_batch_indices(self):
        self.batch_indices = []
        for batch in self.batch_sampler:
            self.batch_indices.append(batch)

        if self.start_iteration is not None:
            self.batch_indices = self.batch_indices[self.start_iteration:]
            self.start_iteration = None

    def __iter__(self):
        if self.batch_indices is None:
            self.setup_batch_indices()
        for batch in self.batch_indices:
            yield batch

        self.batch_indices = None

    def __len__(self):
        return len(self.batch_sampler)
