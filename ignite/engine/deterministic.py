import random
import warnings
from functools import wraps
from typing import Optional, Generator, Callable, Iterator
from collections import OrderedDict

import torch
from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.utils import manual_seed


__all__ = ["update_dataloader", "ReproducibleBatchSampler", "DeterministicEngine"]


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

            from ignite.engine.deterministic import update_dataloader

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


def _repr_rng_state(rng_states):
    from hashlib import md5

    out = " ".join([md5(str(list(s)).encode("utf-8")).hexdigest() for s in rng_states])
    return out


def keep_random_state(func: Callable):
    """Helper decorator to keep random state of torch, numpy and random intact
    while executing a function. For more details on usage, please see
    `"Concepts/Dataflow synchronization" <https://pytorch.org/ignite/concepts.html#dataflow-synchronization>`_.

    Args:
        func (callable): function to decorate
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        rng_states = _get_rng_states()
        func(*args, **kwargs)
        _set_rng_states(rng_states)

    return wrapper


class DeterministicEngine(Engine):
    """Deterministic engine derived from :class:`~ignite.engine.Engine`.

    "Deterministic" run is done by adding additional handlers to synchronize the dataflow and overriding some methods of
    :class:`~ignite.engine.Engine`:

    .. code-block:: python

        for e in range(num_epochs):
            set_seed(seed_offset + e)
            if resume:
                setup_saved_rng_states()
            do_single_epoch_iterations(dataloader)

    If input data provider is `torch.utils.data.DataLoader`, its batch sampler is replaced by
    :class:`~ignite.engine.deterministic.ReproducibleBatchSampler`.

    .. code-block:: python

        for e in range(num_epochs):
            set_seed(seed_offset + e)
            setup_sampling(dataloader)
            if resume:
                setup_saved_rng_states()
            do_single_epoch_iterations(dataloader)

    """

    def __init__(self, process_function: Callable):
        super(DeterministicEngine, self).__init__(process_function)
        self.state_dict_user_keys.append("rng_states")
        self.add_event_handler(Events.STARTED, self._init_run)
        self.add_event_handler(Events.DATALOADER_STOP_ITERATION | Events.TERMINATE_SINGLE_EPOCH, self._setup_seed)

    def state_dict(self) -> OrderedDict:
        state_dict = super(DeterministicEngine, self).state_dict()
        state_dict["rng_states"] = _get_rng_states()
        return state_dict

    def _init_run(self) -> None:
        seed = torch.randint(0, int(1e9), (1,)).item()
        self.state.seed = seed
        if not hasattr(self.state, "rng_states"):
            self.state.rng_states = None

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _setup_engine(self) -> None:
        try:
            self._dataloader_len = None
            if hasattr(self.state.dataloader, "__len__"):
                self._dataloader_len = len(self.state.dataloader)
        except TypeError:
            # _InfiniteConstantSampler can raise a TypeError on DataLoader length of a IterableDataset
            self._dataloader_len = None

        # if input data is torch dataloader we replace batch sampler by a batch sampler
        # such that its random sampling indices are reproducible by prefetching them before data iteration
        if isinstance(self.state.dataloader, torch.utils.data.DataLoader):
            _dataloader_kind = self.state.dataloader._dataset_kind
            if _dataloader_kind == torch.utils.data.dataloader._DatasetKind.Map:
                if (self._dataloader_len is not None) and hasattr(self.state.dataloader.sampler, "epoch"):
                    if self._dataloader_len != self.state.epoch_length:
                        warnings.warn(
                            "When defined engine's epoch length is different of input dataloader length, "
                            "distributed sampler indices can not be setup in a reproducible manner"
                        )

                batch_sampler = self.state.dataloader.batch_sampler
                if not isinstance(batch_sampler, ReproducibleBatchSampler):
                    self.state.dataloader = update_dataloader(
                        self.state.dataloader, ReproducibleBatchSampler(batch_sampler)
                    )

        iteration = self.state.iteration
        self._dataloader_iter = self._from_iteration(iteration)

        # Below we define initial counter value for _run_once_on_dataset to measure a single epoch
        if self.state.epoch_length is not None:
            iteration %= self.state.epoch_length
        self._init_iter.append(iteration)

        # restore rng state
        if getattr(self.state, "rng_states", None) is not None:
            _set_rng_states(self.state.rng_states)
            self.state.rng_states = None

    def _from_iteration(self, iteration: int) -> Iterator:
        data = self.state.dataloader
        if isinstance(data, torch.utils.data.DataLoader):
            try:
                # following is unsafe for IterableDatasets
                iteration %= len(data.batch_sampler)
                # Synchronize dataflow according to state.iteration
                self._setup_seed()
                if iteration > 0:
                    # batch sampler is ReproducibleBatchSampler
                    data.batch_sampler.start_iteration = iteration
                return iter(data)
            except TypeError as e:
                # Probably we can do nothing with DataLoader built upon IterableDatasets
                pass

        self.logger.info("Resuming from iteration for provided data will fetch data until required iteration ...")
        if hasattr(data, "__len__"):
            iteration %= len(data)
        # Synchronize dataflow from the begining
        self._setup_seed(iteration=0)
        data_iter = iter(data)
        counter = 0
        while counter < iteration:
            try:
                next(data_iter)
                counter += 1
            except StopIteration:
                data_iter = iter(data)

        return data_iter

    def _setup_seed(self, _=None, iter_counter=None, iteration=None):
        if iter_counter is None:
            le = self._dataloader_len if self._dataloader_len is not None else 1
        else:
            le = iter_counter
        if iteration is None:
            iteration = self.state.iteration
        manual_seed(self.state.seed + iteration // le)
