import random
import warnings
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, Generator, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.utils import manual_seed

__all__ = ["update_dataloader", "keep_random_state", "ReproducibleBatchSampler", "DeterministicEngine"]


def update_dataloader(dataloader: DataLoader, new_batch_sampler: BatchSampler) -> DataLoader:
    """Helper function to replace current batch sampler of the dataloader by a new batch sampler. Function returns new
    dataloader with new batch sampler.

    Args:
        dataloader: input dataloader
        new_batch_sampler: new batch sampler to use

    Returns:
        DataLoader
    """
    params_keys = [k for k in dataloader.__dict__.keys() if not k.startswith("_")]
    for k in ["batch_size", "sampler", "drop_last", "batch_sampler", "dataset_kind"]:
        if k in params_keys:
            params_keys.remove(k)
    params = {k: getattr(dataloader, k) for k in params_keys}
    params["batch_sampler"] = new_batch_sampler
    return type(dataloader)(**params)


class ReproducibleBatchSampler(BatchSampler):
    """Reproducible batch sampler. This class internally iterates and stores indices of the input batch sampler.
    This helps to start providing data batches from an iteration in a deterministic way.

    Args:
        batch_sampler: batch sampler same as used with `torch.utils.data.DataLoader`.
        start_iteration: optional start iteration.

    Examples:
        Setup dataloader with `ReproducibleBatchSampler` and start providing data batches from an iteration

        .. code-block:: python

            from ignite.engine.deterministic import update_dataloader

            dataloader = update_dataloader(dataloader, ReproducibleBatchSampler(dataloader.batch_sampler))
            # rewind dataloader to a specific iteration:
            dataloader.batch_sampler.start_iteration = start_iteration

    """

    def __init__(self, batch_sampler: BatchSampler, start_iteration: Optional[int] = None):
        if not isinstance(batch_sampler, BatchSampler):
            raise TypeError("Argument batch_sampler should be torch.utils.data.sampler.BatchSampler")

        self.batch_indices = []  # type: List
        self.batch_sampler = batch_sampler
        self.start_iteration = start_iteration
        self.sampler = self.batch_sampler.sampler

    def setup_batch_indices(self) -> None:
        """Setup batch indices."""
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


def _get_rng_states() -> List[Any]:
    output = [random.getstate(), torch.get_rng_state()]
    try:
        import numpy as np

        output.append(np.random.get_state())
    except ImportError:
        pass

    return output


def _set_rng_states(rng_states: List[Any]) -> None:
    random.setstate(rng_states[0])

    if "cpu" not in rng_states[1].device.type:
        rng_states[1] = rng_states[1].cpu()

    torch.set_rng_state(rng_states[1])
    try:
        import numpy as np

        np.random.set_state(rng_states[2])
    except ImportError:
        pass


def _repr_rng_state(rng_states: List[Any]) -> str:
    from hashlib import md5

    out = " ".join([md5(str(list(s)).encode("utf-8")).hexdigest() for s in rng_states])
    return out


def keep_random_state(func: Callable) -> Callable:
    """Helper decorator to keep random state of torch, numpy and random intact
    while executing a function. For more details on usage, please see :ref:`Dataflow synchronization`.

    Args:
        func: function to decorate
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        rng_states = _get_rng_states()
        func(*args, **kwargs)
        _set_rng_states(rng_states)

    return wrapper


class DeterministicEngine(Engine):
    """Deterministic engine derived from :class:`~ignite.engine.engine.Engine`.

    "Deterministic" run is done by adding additional handlers to synchronize the dataflow and overriding some methods of
    :class:`~ignite.engine.engine.Engine`:

    .. code-block:: python

        for e in range(num_epochs):
            set_seed(seed_offset + e)
            if resume:
                setup_saved_rng_states()
            do_single_epoch_iterations(dataloader)

    If input data provider is `DataLoader`, its batch sampler is replaced by
    :class:`~ignite.engine.deterministic.ReproducibleBatchSampler`.

    .. code-block:: python

        for e in range(num_epochs):
            set_seed(seed_offset + e)
            setup_sampling(dataloader)
            if resume:
                setup_saved_rng_states()
            do_single_epoch_iterations(dataloader)

    Internally, `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` are also
    applied.

    For more details about dataflow synchronization, please see :ref:`Dataflow synchronization`.

    .. Note ::

        This class can produce exactly the same dataflow when resuming the run from an epoch (or more precisely from
        dataflow restart) and using torch `DataLoader` with `num_workers > 1` as data provider.

    Args:
        process_function: A function receiving a handle to the engine and the current batch
            in each iteration, and returns data to be stored in the engine's state.
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
        self.state.seed = int(torch.randint(0, int(1e9), (1,)).item())
        if not hasattr(self.state, "rng_states"):
            setattr(self.state, "rng_states", None)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _setup_engine(self) -> None:
        if self.state.dataloader is None:
            raise RuntimeError(
                "Internal error, self.state.dataloader is None. Please, file an issue if you encounter this error."
            )

        self._dataloader_len = self._get_data_length(self.state.dataloader)

        # if input data is torch dataloader we replace batch sampler by a batch sampler
        # such that its random sampling indices are reproducible by prefetching them before data iteration
        if isinstance(self.state.dataloader, DataLoader):
            # attribute _dataset_kind is introduced since 1.3.0 => before 1.3.0 all datasets are map-like
            can_patch_dataloader = True
            if hasattr(self.state.dataloader, "_dataset_kind"):
                from torch.utils.data.dataloader import _DatasetKind

                _dataloader_kind = self.state.dataloader._dataset_kind
                can_patch_dataloader = _dataloader_kind == _DatasetKind.Map
            if can_patch_dataloader:
                if self._dataloader_len is not None and hasattr(self.state.dataloader.sampler, "epoch"):
                    if self._dataloader_len != self.state.epoch_length:
                        warnings.warn(
                            "When defined engine's epoch length is different of input dataloader length, "
                            "distributed sampler indices can not be setup in a reproducible manner"
                        )

                batch_sampler = self.state.dataloader.batch_sampler
                if not (batch_sampler is None or isinstance(batch_sampler, ReproducibleBatchSampler)):
                    self.state.dataloader = update_dataloader(
                        self.state.dataloader, ReproducibleBatchSampler(batch_sampler)  # type: ignore[arg-type]
                    )

        iteration = self.state.iteration
        self._dataloader_iter = self._from_iteration(iteration)

        # Below we define initial counter value for _run_once_on_dataset to measure a single epoch
        if self.state.epoch_length is not None:
            iteration %= self.state.epoch_length
        self._init_iter.append(iteration)

        # restore rng state if in the middle
        in_the_middle = self.state.iteration % self._dataloader_len > 0 if self._dataloader_len is not None else False
        rng_states = getattr(self.state, "rng_states", None)
        if rng_states is not None and in_the_middle:
            _set_rng_states(rng_states)
            setattr(self.state, "rng_states", None)

    def _from_iteration(self, iteration: int) -> Iterator:
        if self.state.dataloader is None:
            raise RuntimeError(
                "Internal error, self.state.dataloader is None. Please, file an issue if you encounter this error."
            )
        data = self.state.dataloader
        if isinstance(data, DataLoader):
            try:
                # following is unsafe for IterableDatasets
                iteration %= len(data.batch_sampler)  # type: ignore[attr-defined, arg-type]
                # Synchronize dataflow according to state.iteration
                self._setup_seed()
                if iteration > 0:
                    # batch sampler is ReproducibleBatchSampler
                    data.batch_sampler.start_iteration = iteration  # type: ignore[attr-defined, union-attr]
                return iter(data)
            except TypeError as e:
                # Probably we can do nothing with DataLoader built upon IterableDatasets
                pass

        self.logger.info("Resuming from iteration for provided data will fetch data until required iteration ...")
        if hasattr(data, "__len__"):
            iteration %= len(data)  # type: ignore[arg-type]
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

    def _setup_seed(self, _: Any = None, iter_counter: Optional[int] = None, iteration: Optional[int] = None) -> None:
        if iter_counter is None:
            le = self._dataloader_len if self._dataloader_len is not None else 1
        elif not iter_counter > 0:
            raise ValueError("iter_counter should be positive value")
        else:
            le = iter_counter
        if iteration is None:
            iteration = self.state.iteration
        manual_seed(self.state.seed + iteration // le)  # type: ignore[operator]
