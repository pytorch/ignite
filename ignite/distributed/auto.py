import warnings
from typing import Any, Callable, Iterator, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler

from ignite.distributed import utils as idist
from ignite.distributed.comp_models import horovod as idist_hvd
from ignite.distributed.comp_models import native as idist_native
from ignite.distributed.comp_models import xla as idist_xla
from ignite.utils import setup_logger

__all__ = ["auto_dataloader", "auto_model", "auto_optim", "DistributedProxySampler"]


def auto_dataloader(dataset: Dataset, **kwargs: Any) -> Union[DataLoader, "_MpDeviceLoader"]:
    """Helper method to create a dataloader adapted for non-distributed and distributed configurations (supporting
    all available backends from :meth:`~ignite.distributed.utils.available_backends()`).

    Internally, we create a dataloader with provided kwargs while applying the following updates:

    - batch size is scaled by world size: ``batch_size / world_size`` if larger or equal world size.
    - number of workers is scaled by number of local processes: ``num_workers / nprocs`` if larger or equal world size.
    - if no sampler provided by user, `torch DistributedSampler`_ is setup.
    - if a sampler is provided by user, it is wrapped by :class:`~ignite.distributed.auto.DistributedProxySampler`.
    - if the default device is 'cuda', `pin_memory` is automatically set to `True`.

    .. warning::

        Custom batch sampler is not adapted for distributed configuration. Please, make sure that provided batch
        sampler is compatible with distributed configuration.

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

    Args:
        dataset (Dataset): input torch dataset
        **kwargs: keyword arguments for `torch DataLoader`_.

    Returns:
        `torch DataLoader`_ or `XLA MpDeviceLoader`_ for XLA devices

    .. _torch DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    .. _XLA MpDeviceLoader: https://github.com/pytorch/xla/blob/master/torch_xla/distributed/parallel_loader.py#L178
    .. _torch DistributedSampler:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
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
            if kwargs.get("sampler", None) is not None:
                sampler = DistributedProxySampler(
                    kwargs["sampler"], num_replicas=world_size, rank=rank
                )  # type: Union[DistributedProxySampler, DistributedSampler, Sampler]
            else:
                sampler = DistributedSampler(
                    dataset, num_replicas=world_size, rank=rank, shuffle=kwargs.get("shuffle", True)
                )
                # we need to remove "shuffle" from kwargs if sampler is used
                if "shuffle" in kwargs:
                    del kwargs["shuffle"]

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

    Examples:

    .. code-block:: python

        import ignite.distribted as idist

        model = idist.auto_model(model)

    In addition with NVidia/Apex, it can be used in the following way:

    .. code-block:: python

        import ignite.distribted as idist

        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        model = idist.auto_model(model)

    Args:
        model (torch.nn.Module): model to adapt.
        sync_bn (bool): if True, applies `torch convert_sync_batchnorm`_ to the model for native torch
            distributed only. Default, False. Note, if using Nvidia/Apex, batchnorm conversion should be
            applied before calling ``amp.initialize``.
        **kwargs: kwargs to model's wrapping class: `torch DistributedDataParallel`_ or `torch DataParallel`_
            if applicable. Please, make sure to use acceptable kwargs for given backend.

    Returns:
        torch.nn.Module

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
        if idist.has_native_dist_support and bnd == idist_native.NCCL:
            if sync_bn:
                logger.info("Convert batch norm to sync batch norm")
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

            if "device_ids" in kwargs:
                raise ValueError(f"Argument kwargs should not contain 'device_ids', but got {kwargs}")

            lrank = idist.get_local_rank()
            logger.info(f"Apply torch DistributedDataParallel on model, device id: {lrank}")
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lrank,], **kwargs)
        elif idist.has_native_dist_support and bnd == idist_native.GLOO:
            if sync_bn:
                logger.info("Convert batch norm to sync batch norm")
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

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


def auto_optim(optimizer: Optimizer) -> Optimizer:
    """Helper method to adapt optimizer for non-distributed and distributed configurations (supporting
    all available backends from :meth:`~ignite.distributed.utils.available_backends()`).

    Internally, this method is no-op for non-distributed and torch native distributed configuration.

    For XLA distributed configuration, we create a new class that inherits from provided optimizer.
    The goal is to override the `step()` method with specific `xm.optimizer_step`_ implementation.

    For Horovod distributed configuration, optimizer is wrapped with Horovod Distributed Optimizer and
    its state is broadcasted from rank 0 to all other processes.

    Examples:

    .. code-block:: python

        import ignite.distributed as idist

        optimizer = idist.auto_optim(optimizer)

    Args:
        optimizer (Optimizer): input torch optimizer

    Returns:
        Optimizer

    .. _xm.optimizer_step: http://pytorch.org/xla/release/1.5/index.html#torch_xla.core.xla_model.optimizer_step

    .. versionchanged:: 0.4.2
        Added Horovod distributed optimizer.
    """
    bnd = idist.backend()
    if idist.has_xla_support and bnd == idist_xla.XLA_TPU:
        cls = type(optimizer.__class__.__name__, (optimizer.__class__,), dict(_XLADistributedOptimizer.__dict__))
        return cls(optimizer)

    if idist.has_hvd_support and bnd == idist_hvd.HOROVOD:
        import horovod.torch as hvd

        optimizer = hvd.DistributedOptimizer(optimizer)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        return optimizer

    return optimizer


class DistributedProxySampler(DistributedSampler):
    """Distributed sampler proxy to adapt user's sampler for distributed data parallelism configuration.

    Code is based on https://github.com/pytorch/pytorch/issues/23430#issuecomment-562350407


    .. note::
        Input sampler is assumed to have a constant size.

    Args:
        sampler (Sampler): Input torch data sampler.
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within ``num_replicas``.

    """

    def __init__(self, sampler: Sampler, num_replicas: Optional[int] = None, rank: Optional[int] = None) -> None:

        if not isinstance(sampler, Sampler):
            raise TypeError(f"Argument sampler should be instance of torch Sampler, but given: {type(sampler)}")

        if not hasattr(sampler, "__len__"):
            raise TypeError("Argument sampler should have length")

        super(DistributedProxySampler, self).__init__(
            sampler, num_replicas=num_replicas, rank=rank, shuffle=False  # type: ignore[arg-type]
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)

        indices = []  # type: List
        while len(indices) < self.total_size:
            indices += list(self.sampler)

        if len(indices) > self.total_size:
            indices = indices[: self.total_size]

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError(f"{len(indices)} vs {self.num_samples}")

        return iter(indices)


if idist.has_xla_support:

    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.parallel_loader import ParallelLoader

    class _MpDeviceLoader:
        # https://github.com/pytorch/xla/pull/2117
        # From pytorch/xla if `torch_xla.distributed.parallel_loader.MpDeviceLoader` is not available
        def __init__(self, loader: Any, device: torch.device, **kwargs: Any) -> None:
            self._loader = loader
            self._device = device
            self._parallel_loader_kwargs = kwargs

        def __iter__(self) -> Iterator:
            parallel_loader = ParallelLoader(self._loader, [self._device], **self._parallel_loader_kwargs)
            return parallel_loader.per_device_loader(self._device)

        def __len__(self) -> int:
            return len(self._loader)

    class _XLADistributedOptimizer(Optimizer):
        def __init__(self, optimizer: Optimizer) -> None:
            super(self.__class__, self).__init__(optimizer.param_groups)  # type: ignore[call-arg]
            self.wrapped_optimizer = optimizer

        def step(self, closure: Optional[Callable] = None) -> None:
            xm.optimizer_step(self.wrapped_optimizer, barrier=True)
