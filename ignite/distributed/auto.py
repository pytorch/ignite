from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler

from ignite.distributed import utils as idist
from ignite.distributed.comp_models import horovod as idist_hvd
from ignite.distributed.comp_models import native as idist_native
from ignite.distributed.comp_models import xla as idist_xla
from ignite.utils import setup_logger

__all__ = ["auto_dataloader", "auto_model", "auto_optim", "DistributedProxySampler"]


def auto_dataloader(dataset: Dataset, **kwargs: Any) -> DataLoader | _MpDeviceLoader:
    """Helper method to create a dataloader adapted for non-distributed and distributed configurations (supporting
    all available backends from :meth:`~ignite.distributed.utils.available_backends()`).

    Internally, we create a dataloader with provided kwargs while applying the following updates:

    - batch size is scaled by world size: ``batch_size / world_size`` if larger or equal world size.
    - number of workers is scaled by number of local processes: ``num_workers / nprocsc` if larger or equal world size.
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

        nproc = idist.get_nprocs_per_node()
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
        if idist.has_native_dist_support and bnd in (idist_native.NCCL, creturn model