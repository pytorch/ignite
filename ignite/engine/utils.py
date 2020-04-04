import inspect

from typing import Optional, Generator, Callable
from functools import wraps
import random

import torch


def update_dataloader(
    dataloader: torch.utils.data.DataLoader, new_batch_sampler: torch.utils.data.sampler.BatchSampler
) -> torch.utils.data.DataLoader:
    """Helper function to replace current batch sampler of the dataloader by a new batch sampler. Function returns new
    dataloader with new batch sampler.

    Args:
        dataloader (torch.utils.data.DataLoader): input dataloader
        new_batch_sampler (torch.utils.data.sampler.BatchSampler): new batch sampler to use

    Returns:
        torch.utils.data.DataLoader
    """
    params_keys = [k for k in dataloader.__dict__.keys() if not k.startswith("_")]
    for k in ["batch_size", "sampler", "drop_last", "batch_sampler", "dataset_kind"]:
        if k in params_keys:
            params_keys.remove(k)
    params = {k: getattr(dataloader, k) for k in params_keys}
    params["batch_sampler"] = new_batch_sampler
    return type(dataloader)(**params)


class ReproducibleBatchSampler(torch.utils.data.sampler.BatchSampler):
    """Reproducible batch sampler. This class internally iterates and stores indices of the input batch sampler.
    This helps to start providing data batches from an iteration in a deterministic way.

    Usage:

        Setup dataloader with `ReproducibleBatchSampler` and start providing data batches from an iteration:

        .. code-block:: python

            from ignite.engine.utils import update_dataloader

            dataloader = update_dataloader(dataloader, ReproducibleBatchSampler(dataloader.batch_sampler))
            # rewind dataloader to a specific iteration:
            dataloader.batch_sampler.start_iteration = start_iteration

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
            self.batch_indices = self.batch_indices[self.start_iteration :]
            self.start_iteration = None

    def __iter__(self) -> Generator:
        self.setup_batch_indices()
        for batch in self.batch_indices:
            yield batch

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
        raise ValueError(
            "Error adding {} '{}': "
            "takes parameters {} but will be called with {} "
            "({}).".format(fn, fn_description, fn_params, passed_params, exception_msg)
        )


def _get_rng_states():
    output = [random.getstate(), torch.get_rng_state()]
    try:
        import numpy as np

        output.append(np.random.get_state())
    except ImportError:
        pass

    return output


def _set_rng_states(rng_states):
    random.setstate(rng_states[0])
    torch.set_rng_state(rng_states[1])
    try:
        import numpy as np

        np.random.set_state(rng_states[2])
    except ImportError:
        pass


def keep_random_state(func):
    """Helper decorator to keep random state of torch, numpy and random intact
    while executing a function. For more details on usage, please see
    `"Concepts/Random state synchronization" <https://pytorch.org/ignite/concepts.html#random-state-synchronization>`_.

    Args:
        func (callable): function to decorate
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        rng_states = _get_rng_states()
        func(*args, **kwargs)
        _set_rng_states(rng_states)

    return wrapper
