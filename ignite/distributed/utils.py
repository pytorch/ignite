from typing import Optional, Union, Tuple
import socket
from functools import wraps

import torch
import torch.distributed as dist

try:
    import torch_xla.core.xla_model as xm

    has_xla_support = True
except ImportError:
    has_xla_support = False

from ignite.utils import setup_logger
from ignite.distributed.comp_models import registered_computation_models, _SerialModel


# default: _SerialModel
_model = _SerialModel


def _sync_model_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if isinstance(_model, _SerialModel):
            _sync_model()

        return func(*args, **kwargs)

    return wrapper


def _sync_model():
    global _model


@_sync_model_wrapper
def device() -> Union[torch.device, str]:
    return _model.device()


@_sync_model_wrapper
def backend() -> Optional[str]:
    """Returns computation model's backend.
    For non-distributed model, the backend is None.
    For native torch distributed model, the backend can be "nccl", "gloo", "mpi"
    For XLA distributed model, the backend can be "xla-tpu"

    Returns:
        str or None
    """
    return _model.backend()


def available_backends() -> Tuple[str]:
    out = ()
    for m in registered_computation_models:
        out += m.available_backends
    return out


@_sync_model_wrapper
def model_name() -> str:
    return _model.name


@_sync_model_wrapper
def is_distributed() -> bool:
    return _model.is_distributed()


# @_sync_model_wrapper
# def is_initialized() -> bool:
#     return _model.is_initialized()


@_sync_model_wrapper
def get_world_size() -> int:
    return _model.get_world_size()


@_sync_model_wrapper
def get_rank() -> int:
    return _model.get_rank()


@_sync_model_wrapper
def get_local_rank() -> int:
    return _model.get_local_rank()


# experimental
# to be tested with slurm
# it depends on how scheduling was done


@_sync_model_wrapper
def get_ntasks_per_node() -> int:
    return _model.get_ntasks_per_node()


@_sync_model_wrapper
def get_num_nodes() -> int:
    return _model.get_num_nodes()


@_sync_model_wrapper
def get_node_index() -> int:
    return _model.get_rank() % _model.get_ntasks_per_node()


def hostname() -> str:
    return socket.gethostname()


def spawn(fn, args, backend, num_workers_per_machine, num_machines, machine_rank, **kwargs):
    pass
    # !!! WE NEED TO SETUP _model !!!

    # _assert_backend(backend)
    # for comp_model_cls in registered_computation_models:
    #     if backend not in comp_model_cls.available_backends:
    #         continue
    #     comp_model_cls.spawn(
    #         fn,
    #         args=args,
    #         num_workers_per_machine=num_workers_per_machine,
    #         num_machines=num_machines,
    #         machine_rank=machine_rank,
    #         **kwargs
    #     )


def _find_best_dist_backend():
    assert dist.is_available()
    # select best backend nccl first, gloo second
    if dist.is_nccl_available():
        return "nccl"
    elif dist.is_gloo_available():
        return "gloo"
    else:
        raise Exception("No backend found (missing gloo and nccl")


def _sanity_check():
    # assert _model.is_initialized()
    assert _model.get_world_size() == _model.get_nnodes() * _model.get_ntasks_per_node()
    assert _model.get_local_rank() < _model.get_ntasks_per_node()
    assert _model.get_rank() < _model.get_world_size()
    assert _model.get_node() < _model.get_nnodes()


def _assert_backend(backend):
    backends = available_backends()
    if backend not in backends:
        raise ValueError("Backend should be one of '{}'".format(backends))


def initialize(backend: Optional[str] = None, timeout: Optional["timedelta"] = None, **kwargs):
    """

    Args:
        backend (str, optional): backend to initialize computation model.
        timeout (timedelta, optional): process group initialization timeout. Applied to `torch.distributed`.

    Returns:

    """
    global _model

    if not (has_xla_support or dist.is_available()):
        # nothing to do => serial model
        # maybe warn about this
        return

    if backend is None:
        _sync_model()
        return

    _assert_backend(backend)

    for comp_model_cls in registered_computation_models:
        if backend not in comp_model_cls.available_backends:
            continue
        _model = comp_model_cls(backend, timeout=timeout, **kwargs)

    # if has_xla_support:
    #     # no backend for tpu
    #     assert backend is None
    #     # if device imposed, it should be tpu
    #     if device is not None:
    #         assert device == "tpu"
    #     _device = "tpu"
    #     _model = _XlaModel()
    #     # how to check if already initialized ?
    #     xm.rendezvous("init")
    # elif dist.is_available():
    #     if backend is None:
    #         _backend = _find_best_dist_backend()
    #     else:
    #         # imposed by user
    #         _backend = backend
    #     assert not dist.is_initialized()
    #     if backend == "nccl":
    #         torch.backends.cudnn.benchmark = True
    #     _model = _DistModel()
    #     # should this be in DistModel ?
    #     dist.init_process_group(backend, init_method="env://")
    #     # compute missing information (nnodes, tasks_per_node, etc.)
    #     _model.finalize_init()
    #     if device() == "cuda":
    #         torch.cuda.set_device(get_local_rank())
    #         # number of proc per node should be related to number of GPUs
    #         assert get_ntasks_per_node() <= torch.cuda.device_count()
    #     # TODO: check if backend is available
    #     # TODO: check device wrt backend (nccl and cpu is not possible)
    # # test the configuration
    # _sanity_check()


def finalize():
    _model.finalize()


def show_config():

    # setup parallel logger
    logger = setup_logger(__name__)

    logger.info("model = {}".format(model_name()))
    logger.info("is distributed = {}".format(is_distributed()))
    logger.info("backend = {}".format(backend()))
    logger.info("device = {}".format(device()))
    logger.info("hostname = {}".format(hostname()))
    logger.info("world size = {}".format(get_world_size()))
    logger.info("rank = {}".format(get_rank()))
    logger.info("ntasks_per_node = {}".format(get_ntasks_per_node()))
    logger.info("nnodes = {}".format(get_nnodes()))
    logger.info("node = {}".format(get_node_index()))
