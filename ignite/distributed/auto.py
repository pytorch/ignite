from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler, Sampler

from ignite.distributed import utils as idist
from ignite.distributed.comp_models import horovod as idist_hvd
from ignite.distributed.comp_models import native as idist_native
from ignite.distributed.comp_models import xla as idist_xla
from ignite.utils import setup_logger

__all__ = ["auto_dataloader", "auto_model", "auto_optim", "DistributedProxySampler", "StratifiedBatchSampler"]


def auto_dataloader(dataset: Dataset, **kwargs: Any) -> DataLoader | _MpDeviceLoader:
    """Helper method to create a dataloader adapted for non-distributed and distributed configurations (supporting
    all available backends from :meth:`~ignite.distributed.utils.available_backends()`).

    Internally, we create a dataloader with provided kwargs while applying the following updates:

    - batch size is scaled by world size: ``batch_size / world_size`` if larger or equal world size.
    - number of workers is scaled by number of local processes: ``num_workers / nprocs`` if larger or equal world size.
    - if no sampler provided by user, a `torch DistributedSampler`_ is setup.
    - if a `torch DistributedSampler`_ is provided by user, it is used without wrapping it.
    - if another sampler is provided, it is wrapped by :class:`~ignite.distributed.auto.DistributedProxySampler`.
    - if the default device is 'cuda', `pin_memory` is automatically set to `True`.

    .. warning::

        Custom batch sampler is not adapted for distributed configuration. Please, make sure that provided batch
        sampler is compatible with distributed configuration.

    Args:
        dataset: input torch dataset. If input dataset is `torch IterableDataset`_ then dataloader will be
            created without any distributed sampling. Please, make sure that the dataset itself produces
            different data on different ranks.
        kwargs: keyword arguments for `torch DataLoader`_.

    Returns:
        `torch DataLoader`_ or `XLA MpDeviceLoader`_ for XLA devices

    Examples:
        .. code-block:: python

            import ignite.distribted as idist

            train_loader = idist.auto_dataloader(
                train_dataset,
                batch_size=32,
                num_workers=4,
                shuffle=True,
                pin_memory="cuda" in idist.device().type,
                drop_last=True,
            )

    .. _torch DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    .. _XLA MpDeviceLoader:
        https://pytorch.org/xla/release/2.0/index.html#running-on-multiple-xla-devices-with-multi-processing
    .. _torch DistributedSampler:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    .. _torch IterableDataset: https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """
    rank = idist.get_rank()
    world_size = idist.get_world_size()

    logger = setup_logger(__name__ + ".auto_dataloader")
    if world_size > 1:
        if "batch_size" in kwargs and kwargs["batch_size"] >= world_size:
            kwargs["batch_size"] //= world_size

        nproc = idist.get_nproc_per_node()
        if "num_workers" in kwargs and kwargs["num_workers"] >= nproc:
            kwargs["num_workers"] = (kwargs["num_workers"] + nproc - 1) // nproc

        if "batch_sampler" not in kwargs:
            if isinstance(dataset, IterableDataset):
                logger.info(
                    "Found iterable dataset, dataloader will be created without any distributed sampling. "
                    "Please, make sure that the dataset itself produces different data on different ranks."
                )
            else:
                sampler: DistributedProxySampler | DistributedSampler | Sampler | None
                sampler = kwargs.get("sampler")
                if isinstance(sampler, DistributedSampler):
                    if sampler.rank != rank:
                        warnings.warn(f"Found distributed sampler with rank={sampler.rank}, but process rank is {rank}")
                    if sampler.num_replicas != world_size:
                        warnings.warn(
                            f"Found distributed sampler with num_replicas={sampler.num_replicas}, "
                            f"but world size is {world_size}"
                        )
                elif sampler is None:
                    # removes "shuffle" from kwargs if sampler is used
                    shuffle = kwargs.pop("shuffle", True)
                    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
                else:
                    sampler = DistributedProxySampler(sampler, num_replicas=world_size, rank=rank)
                kwargs["sampler"] = sampler
        else:
            warnings.warn(
                "Found batch_sampler in provided kwargs. Please, make sure that it is compatible "
                "with distributed configuration"
            )

    if idist.has_xla_support and idist.backend() == idist_xla.XLA_TPU and kwargs.get("pin_memory", False):
        # TODO: How about XLA GPU ?
        warnings.warn(
            "Found incompatible options: xla support and pin_memory args equal True. "
            "Argument `pin_memory=False` will be used to construct data loader."
        )
        kwargs["pin_memory"] = False
    else:
        kwargs["pin_memory"] = kwargs.get("pin_memory", "cuda" in idist.device().type)

    logger.info(f"Use data loader kwargs for dataset '{repr(dataset)[:20].strip()}': \n\t{kwargs}")
    dataloader = DataLoader(dataset, **kwargs)

    if idist.has_xla_support and idist.backend() == idist_xla.XLA_TPU and world_size > 1:
        logger.info("DataLoader is wrapped by `MpDeviceLoader` on XLA")

        mp_device_loader_cls = _MpDeviceLoader
        try:
            from torch_xla.distributed.parallel_loader import MpDeviceLoader

            mp_device_loader_cls = MpDeviceLoader
        except ImportError:
            pass

        mp_dataloader = mp_device_loader_cls(dataloader, idist.device())
        mp_dataloader.sampler = dataloader.sampler  # type: ignore[attr-defined]
        return mp_dataloader

    return dataloader


def auto_model(model: nn.Module, sync_bn: bool = False, **kwargs: Any) -> nn.Module:
    """Helper method to adapt provided model for non-distributed and distributed configurations (supporting
    all available backends from :meth:`~ignite.distributed.utils.available_backends()`).

    Internally, we perform to following:

    - send model to current :meth:`~ignite.distributed.utils.device()` if model's parameters are not on the device.
    - wrap the model to `torch DistributedDataParallel`_ for native torch distributed if world size is larger than 1.
    - wrap the model to `torch DataParallel`_ if no distributed context found and more than one CUDA devices available.
    - broadcast the initial variable states from rank 0 to all other processes if Horovod distributed framework is used.

    Args:
        model: model to adapt.
        sync_bn: if True, applies `torch convert_sync_batchnorm`_ to the model for native torch
            distributed only. Default, False. Note, if using Nvidia/Apex, batchnorm conversion should be
            applied before calling ``amp.initialize``.
        kwargs: kwargs to model's wrapping class: `torch DistributedDataParallel`_ or `torch DataParallel`_
            if applicable. Please, make sure to use acceptable kwargs for given backend.

    Returns:
        torch.nn.Module

    Examples:
        .. code-block:: python

            import ignite.distribted as idist

            model = idist.auto_model(model)

        In addition with NVidia/Apex, it can be used in the following way:

        .. code-block:: python

            import ignite.distribted as idist

            model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
            model = idist.auto_model(model)

    .. _torch DistributedDataParallel: https://pytorch.org/docs/stable/generated/torch.nn.parallel.
        DistributedDataParallel.html
    .. _torch DataParallel: https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
    .. _torch convert_sync_batchnorm: https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#
        torch.nn.SyncBatchNorm.convert_sync_batchnorm

    .. versionchanged:: 0.4.2

        - Added Horovod distributed framework.
        - Added ``sync_bn`` argument.

    .. versionchanged:: 0.4.3
        Added kwargs to ``idist.auto_model``.
    """
    logger = setup_logger(__name__ + ".auto_model")

    # Put model's parameters to device if its parameters are not on the device
    device = idist.device()
    if not all([p.device == device for p in model.parameters()]):
        model.to(device)

    # distributed data parallel model
    if idist.get_world_size() > 1:
        bnd = idist.backend()
        if idist.has_native_dist_support and bnd in (idist_native.NCCL, idist_native.GLOO, idist_native.MPI):
            if sync_bn:
                logger.info("Convert batch norm to sync batch norm")
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

            if torch.cuda.is_available():
                if "device_ids" in kwargs:
                    raise ValueError(f"Argument kwargs should not contain 'device_ids', but got {kwargs}")

                lrank = idist.get_local_rank()
                logger.info(f"Apply torch DistributedDataParallel on model, device id: {lrank}")
                kwargs["device_ids"] = [
                    lrank,
                ]
            else:
                logger.info("Apply torch DistributedDataParallel on model")

            model = torch.nn.parallel.DistributedDataParallel(model, **kwargs)
        elif idist.has_hvd_support and bnd == idist_hvd.HOROVOD:
            import horovod.torch as hvd

            logger.info("Broadcast the initial variable states from rank 0 to all other processes")
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # not distributed but multiple GPUs reachable so data parallel model
    elif torch.cuda.device_count() > 1 and "cuda" in idist.device().type:
        logger.info("Apply torch DataParallel on model")
        model = torch.nn.parallel.DataParallel(model, **kwargs)

    return model


def auto_optim(optimizer: Optimizer, **kwargs: Any) -> Optimizer:
    """Helper method to adapt optimizer for non-distributed and distributed configurations (supporting
    all available backends from :meth:`~ignite.distributed.utils.available_backends()`).

    Internally, this method is no-op for non-distributed and torch native distributed configuration.

    For XLA distributed configuration, we create a new class that inherits from provided optimizer.
    The goal is to override the `step()` method with specific `xm.optimizer_step`_ implementation.

    For Horovod distributed configuration, optimizer is wrapped with Horovod Distributed Optimizer and
    its state is broadcasted from rank 0 to all other processes.

    Args:
        optimizer: input torch optimizer
        kwargs: kwargs to Horovod backend's DistributedOptimizer.

    Returns:
        Optimizer

    Examples:
        .. code-block:: python

            import ignite.distributed as idist

            optimizer = idist.auto_optim(optimizer)

    .. _xm.optimizer_step: https://pytorch.org/xla/release/1.5/index.html#torch_xla.core.xla_model.optimizer_step

    .. versionchanged:: 0.4.2
        Added Horovod distributed optimizer.

    .. versionchanged:: 0.4.7
        Added kwargs to ``idist.auto_optim``.
    """
    bnd = idist.backend()
    if idist.has_xla_support and bnd == idist_xla.XLA_TPU:
        cls = type(optimizer.__class__.__name__, (optimizer.__class__,), dict(_XLADistributedOptimizer.__dict__))
        return cls(optimizer)

    if idist.has_hvd_support and bnd == idist_hvd.HOROVOD:
        import horovod.torch as hvd

        optimizer = hvd.DistributedOptimizer(optimizer, **kwargs)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        return optimizer

    return optimizer


class DistributedProxySampler(DistributedSampler):
    """Distributed sampler proxy to adapt user's sampler for distributed data parallelism configuration.

    Code is based on https://github.com/pytorch/pytorch/issues/23430#issuecomment-562350407

    Args:
        sampler: Input torch data sampler.
        num_replicas: Number of processes participating in distributed training.
        rank: Rank of the current process within ``num_replicas``.

    .. note::
        Input sampler is assumed to have a constant size.
    """

    def __init__(self, sampler: Sampler, num_replicas: int | None = None, rank: int | None = None) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError(f"Argument sampler should be instance of torch Sampler, but given: {type(sampler)}")

        if isinstance(sampler, DistributedSampler):
            raise TypeError("Argument sampler must not be a distributed sampler already")

        if not hasattr(sampler, "__len__"):
            raise TypeError("Argument sampler should have length")

        super().__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=False)  # type: ignore[arg-type]
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)

        indices: list = []
        while len(indices) < self.total_size:
            indices += list(self.sampler)

        if len(indices) > self.total_size:
            indices = indices[: self.total_size]

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError(f"{len(indices)} vs {self.num_samples}")

        return iter(indices)


class StratifiedBatchSampler(BatchSampler):
    """Batch sampler guaranteeing that every group is represented in every batch.

    Unlike `torch WeightedRandomSampler`_, which only makes rare groups more likely, this sampler puts a fixed
    number of samples of each group into each batch. This is required by methods where a missing group zeroes out
    a loss term, e.g. Group Distributionally Robust Optimization or contrastive learning.

    It is a batch sampler and should be passed to ``batch_sampler`` of a dataloader, not to ``sampler``:

    .. code-block:: python

        batch_sampler = StratifiedBatchSampler(group_ids, batch_size=32)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

        @trainer.on(Events.EPOCH_STARTED)
        def set_epoch(engine):
            batch_sampler.set_epoch(engine.state.epoch)

    Args:
        labels: group id of every dataset sample, as a sequence, numpy array or 1-D tensor. Groups are derived from
            the distinct values, which can be of any hashable type.
        batch_size: number of samples per batch. Should be larger or equal to the number of groups. Every yielded
            batch has exactly this size. If it is not divisible by the number of groups, the leftover slots are
            given to a rotating subset of the groups.
        strategy: if ``"oversample"`` (default), an epoch lasts until the largest group is exhausted and smaller
            groups are cycled, such that no sample is discarded. If ``"undersample"``, an epoch stops once the
            smallest group is exhausted and larger groups are truncated. Groups are reshuffled at every epoch, so
            the truncated part of a larger group is different at every epoch and no sample is permanently unused.
        num_replicas: number of processes participating in distributed training. Defaults to
            :meth:`~ignite.distributed.utils.get_world_size`.
        rank: rank of the current process within ``num_replicas``. Defaults to
            :meth:`~ignite.distributed.utils.get_rank`.
        seed: random seed used to shuffle the groups. The seed of an epoch is ``seed + epoch``.

    .. note::
        In a distributed configuration, groups are sharded across ranks individually, so that every batch of every
        rank still contains every group. Wrapping this sampler with
        :class:`~ignite.distributed.auto.DistributedProxySampler` or `torch DistributedSampler`_ instead would shard
        the flat index list and could leave a rank without any sample of a rare group. Each group should therefore
        have at least
        ``num_replicas`` samples. Groups are truncated to a multiple of ``num_replicas``, so that all ranks yield
        the same number of batches.

    .. note::
        As for `torch DistributedSampler`_, :meth:`set_epoch` should be called at the beginning of every epoch,
        otherwise the same batches are produced at every epoch.

    .. _torch WeightedRandomSampler:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler
    .. _torch DistributedSampler:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler

    .. versionadded:: 0.6.0
    """

    def __init__(
        self,
        labels: Any,
        batch_size: int,
        strategy: str = "oversample",
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
    ) -> None:
        if strategy not in ("oversample", "undersample"):
            raise ValueError(f"Argument strategy should be 'oversample' or 'undersample', but given {strategy}")

        ndim = getattr(labels, "ndim", 1)
        if ndim != 1:
            raise ValueError(f"Argument labels should be one-dimensional, but given {ndim} dimensions")
        if hasattr(labels, "tolist"):
            labels = labels.tolist()

        groups: dict[Any, list[int]] = {}
        for index, label in enumerate(labels):
            groups.setdefault(label, []).append(index)
        if len(groups) == 0:
            raise ValueError("Argument labels should not be empty")

        self.groups = list(groups.values())
        num_groups = len(self.groups)
        if batch_size < num_groups:
            raise ValueError(
                f"Argument batch_size should be larger or equal to the number of groups ({num_groups}), "
                f"but given {batch_size}"
            )

        self.num_replicas = idist.get_world_size() if num_replicas is None else num_replicas
        self.rank = idist.get_rank() if rank is None else rank
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"Argument rank should be in [0, {self.num_replicas}), but given {self.rank}")

        # all ranks drop the same tail, otherwise they yield a different number of batches and collective ops hang
        self.shard_sizes = [len(group) // self.num_replicas for group in self.groups]
        if min(self.shard_sizes) == 0:
            smallest = min(len(group) for group in self.groups)
            raise ValueError(
                f"Smallest group has {smallest} samples, less than num_replicas={self.num_replicas}. "
                "Every group should have at least one sample per rank"
            )

        self.samples_per_group = batch_size // num_groups
        self.leftover_slots = batch_size % num_groups
        if strategy == "oversample":
            self.num_batches = max(-(-size // self.samples_per_group) for size in self.shard_sizes)
        else:
            self.num_batches = max(1, min(size // self.samples_per_group for size in self.shard_sizes))

        self.strategy = strategy
        self.seed = seed
        self.epoch = 0
        super().__init__(sampler=None, batch_size=batch_size, drop_last=False)  # type: ignore[arg-type]

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed the shuffling of the groups.

        Args:
            epoch: epoch number.
        """
        self.epoch = epoch

    def _cycle(self, shard: list[int], generator: torch.Generator) -> Iterator[int]:
        # reshuffled on every pass, unlike itertools.cycle which replays a frozen order
        while True:
            for i in torch.randperm(len(shard), generator=generator).tolist():
                yield shard[i]

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)

        cycles = []
        for group, size in zip(self.groups, self.shard_sizes):
            perm = torch.randperm(len(group), generator=generator).tolist()
            shuffled = [group[i] for i in perm]
            # same permutation on every rank, so that the strided shards are disjoint
            cycles.append(self._cycle(shuffled[self.rank :: self.num_replicas][:size], generator))

        offset = 0
        for _ in range(self.num_batches):
            # rotated, such that no group systematically gets the slots left over by the integer division
            bonus = {(offset + i) % len(cycles) for i in range(self.leftover_slots)}
            offset += self.leftover_slots

            batch = []
            for i, cycle in enumerate(cycles):
                for _ in range(self.samples_per_group + (1 if i in bonus else 0)):
                    batch.append(next(cycle))

            yield [batch[i] for i in torch.randperm(len(batch), generator=generator).tolist()]

    def __len__(self) -> int:
        return self.num_batches


if idist.has_xla_support:
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.parallel_loader import ParallelLoader

    class _MpDeviceLoader:
        # https://github.com/pytorch/xla/pull/2117
        # From pytorch/xla if `torch_xla.distributed.parallel_loader.MpDeviceLoader` is not available
        def __init__(self, loader: Any, device: torch.device, **kwargs: Any) -> None:
            self._loader = loader
            # pyrefly: ignore [read-only]
            self._device = device
            self._parallel_loader_kwargs = kwargs

        def __iter__(self) -> Iterator:
            parallel_loader = ParallelLoader(self._loader, [self._device], **self._parallel_loader_kwargs)
            return parallel_loader.per_device_loader(self._device)

        def __len__(self) -> int:
            return len(self._loader)

    class _XLADistributedOptimizer(Optimizer):
        def __init__(self, optimizer: Optimizer) -> None:
            super().__init__(optimizer.param_groups)  # type: ignore[call-arg]
            self.wrapped_optimizer = optimizer

        def step(self, closure: Any = None) -> Any:
            xm.optimizer_step(self.wrapped_optimizer, barrier=True)
