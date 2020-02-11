import inspect

from typing import Optional, Generator, Callable
import torch


def _update_dataloader(dataloader: torch.utils.data.DataLoader,
                       new_batch_sampler: torch.utils.data.sampler.BatchSampler) -> torch.utils.data.DataLoader:
    params_keys = [k for k in dataloader.__dict__.keys() if not k.startswith("_")]
    for k in ['batch_size', 'sampler', 'drop_last', 'batch_sampler', 'dataset_kind']:
        if k in params_keys:
            params_keys.remove(k)
    params = {k: getattr(dataloader, k) for k in params_keys}
    params['batch_sampler'] = new_batch_sampler
    return type(dataloader)(**params)


class ReproducibleBatchSampler(torch.utils.data.sampler.BatchSampler):
    """Reproducible batch sampler. Internally, this class iterates and stores indices of the input batch sampler.

    Args:
        batch_sampler (torch.utils.data.sampler.BatchSampler): batch sampler same as used with
            `torch.utils.data.DataLoader`
        start_iteration (int, optional): optional start iteration
    """

    def __init__(self, batch_sampler: torch.utils.data.sampler.BatchSampler, start_iteration: Optional[int] = None):
        if not isinstance(batch_sampler, torch.utils.data.sampler.BatchSampler):
            raise TypeError("Argument batch_sampler should be torch.utils.data.sampler.BatchSampler")

        self.batch_indices = None
        self.batch_sampler = batch_sampler
        self.start_iteration = start_iteration
        self.sampler = self.batch_sampler.sampler

    def setup_batch_indices(self) -> None:
        self.batch_indices = []
        for batch in self.batch_sampler:
            self.batch_indices.append(batch)

        if self.start_iteration is not None:
            self.batch_indices = self.batch_indices[self.start_iteration:]
            self.start_iteration = None

    def __iter__(self) -> Generator:
        if self.batch_indices is None:
            self.setup_batch_indices()
        for batch in self.batch_indices:
            yield batch

        self.batch_indices = None

    def __len__(self) -> int:
        return len(self.batch_sampler)


def _check_signature(engine, fn: Callable, fn_description: str, *args, **kwargs) -> None:

    signature = inspect.signature(fn)
    try:
        signature.bind(engine, *args, **kwargs)
    except TypeError as exc:
        fn_params = list(signature.parameters)
        exception_msg = str(exc)
        passed_params = [engine] + list(args) + list(kwargs)
        raise ValueError("Error adding {} '{}': "
                         "takes parameters {} but will be called with {} "
                         "({}).".format(fn, fn_description, fn_params, passed_params, exception_msg))
