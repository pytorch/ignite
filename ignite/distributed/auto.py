import warnings

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler

from ignite.distributed import utils as idist
from ignite.distributed.comp_models import native as idist_native
from ignite.distributed.comp_models import xla as idist_xla
from ignite.utils import setup_logger

__all__ = ["auto_dataloader", "auto_model", "auto_optim", "DistributedProxySampler"]


def auto_dataloader(dataset, **kwargs):
    """Helper method to create a dataloader adapted for non-distributed and distributed configurations (supporting
    all available backends from :meth:`~ignite.distributed.utils.available_backends()`).

    Internally, we create a dataloader with provided kwargs while applying the following updates:

    - batch size is scaled by world size: ``batch_size / world_size`` if larger or equal world size.
    - number of workers is scaled by number of local processes: ``num_workers / nprocs`` if larger or equal world size.
    - if no sampler provided by user, `torch DistributedSampler` is setup.
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
                sampler = DistributedProxySampler(kwargs["sampler"], num_replicas=world_size, rank=rank)
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

    logger.info("Use data loader kwargs for dataset '{}': \n\t{}".format(repr(dataset)[:20].strip(), kwargs))
    dataloader = DataLoader(dataset, **kwargs)

    if idist.has_xla_support and idist.backend() == idist_xla.XLA_TPU and world_size > 1:

        logger.info("DataLoader is wrapped by `MpDeviceLoader` on XLA")

        mp_device_loader_cls = _MpDeviceLoader
        try:
            from torch_xla.distributed.parallel_loader import MpDeviceLoader

            mp_device_loader_cls = MpDeviceLoader
        except ImportError:
            pass

        sampler = dataloader.sampler
        dataloader = mp_device_loader_cls(dataloader, idist.device())
        dataloader.sampler = sampler

    return dataloader


def auto_model(model: nn.Module) -> nn.Module:
    """Helper method to adapt provided model for non-distributed and distributed configurations (supporting
    all available backends from :meth:`~ignite.distributed.utils.available_backends()`).

    Internally, we perform to following:

    - send model to current :meth:`~ignite.distributed.utils.device()`.
    - wrap the model to `torch DistributedDataParallel`_ for native torch distributed if world size is larger than 1
    - wrap the model to `torch DataParallel`_ if no distributed context found and more than one CUDA devices available.

    Examples:

    .. code-block:: python

        import ignite.distribted as idist

        model = idist.auto_model(model)

    Args:
        model (torch.nn.Module): model to adapt.

    Returns:
        torch.nn.Module

    .. _torch DistributedDataParallel: https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel
    .. _torch DataParallel: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
    """
    logger = setup_logger(__name__ + ".auto_model")

    model.to(idist.device())

    # distributed data parallel model
    if idist.get_world_size() > 1:
        if idist.backend() == idist_native.NCCL:
            lrank = idist.get_local_rank()
            logger.info("Apply torch DistributedDataParallel on model, device id: {}".format(lrank))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lrank,])
        elif idist.backend() == idist_native.GLOO:
            logger.info("Apply torch DistributedDataParallel on model")
            model = torch.nn.parallel.DistributedDataParallel(model)

    # not distributed but multiple GPUs reachable so data parallel model
    elif torch.cuda.device_count() > 1 and "cuda" in idist.device().type:
        logger.info("Apply torch DataParallel on model")
        model = torch.nn.parallel.DataParallel(model)

    return model


def auto_optim(optimizer: Optimizer) -> Optimizer:
    """Helper method to adapt optimizer for non-distributed and distributed configurations (supporting
    all available backends from :meth:`~ignite.distributed.utils.available_backends()`).

    Internally, this method is no-op for non-distributed and torch native distributed configuration.
    For XLA distributed configuration, we create a new class that inherits from provided optimizer.
    The goal is to override the `step()` method with specific `xm.optimizer_step`_ implementation.

    Examples:

    .. code-block:: python

        import ignite.distribted as idist

        optimizer = idist.auto_optim(optimizer)


    Args:
        optimizer (Optimizer): input torch optimizer

    Returns:
        Optimizer

    .. _xm.optimizer_step: http://pytorch.org/xla/release/1.5/index.html#torch_xla.core.xla_model.optimizer_step

    """
    if not (idist.has_xla_support and idist.backend() == idist_xla.XLA_TPU):
        return optimizer

    cls = type(optimizer.__class__.__name__, (optimizer.__class__,), dict(_XLADistributedOptimizer.__dict__))
    return cls(optimizer)


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

    def __init__(self, sampler: Sampler, num_replicas=None, rank=None):

        if not isinstance(sampler, Sampler):
            raise TypeError("Argument sampler should be instance of torch Sampler, but given: {}".format(type(sampler)))

        if not hasattr(sampler, "__len__"):
            raise TypeError("Argument sampler should have length")

        super(DistributedProxySampler, self).__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)

        indices = []
        while len(indices) < self.total_size:
            indices += list(self.sampler)

        if len(indices) > self.total_size:
            indices = indices[: self.total_size]

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)


if idist.has_xla_support:

    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.parallel_loader import ParallelLoader

    class _MpDeviceLoader:
        # https://github.com/pytorch/xla/pull/2117
        # From pytorch/xla if `torch_xla.distributed.parallel_loader.MpDeviceLoader` is not available
        def __init__(self, loader, device, **kwargs):
            self._loader = loader
            self._device = device
            self._parallel_loader_kwargs = kwargs

        def __iter__(self):
            parallel_loader = ParallelLoader(self._loader, [self._device], **self._parallel_loader_kwargs)
            return parallel_loader.per_device_loader(self._device)

        def __len__(self):
            return len(self._loader)

    class _XLADistributedOptimizer(Optimizer):
        def __init__(self, optimizer):
            super(self.__class__, self).__init__(optimizer.param_groups)
            self.wrapped_optimizer = optimizer

        def step(self, closure=None):
            xm.optimizer_step(self.wrapped_optimizer, barrier=True)
