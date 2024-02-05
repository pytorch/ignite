import socket
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

import torch

from ignite.distributed.comp_models import (
    _SerialModel,
    has_hvd_support,
    has_native_dist_support,
    has_xla_support,
    registered_computation_models,
)
from ignite.utils import setup_logger

__all__ = [
    "backend",
    "broadcast",
    "device",
    "available_backends",
    "model_name",
    "get_world_size",
    "get_rank",
    "get_local_rank",
    "get_nproc_per_node",
    "get_node_rank",
    "get_nnodes",
    "spawn",
    "initialize",
    "finalize",
    "show_config",
    "set_local_rank",
    "all_reduce",
    "all_gather",
    "barrier",
    "hostname",
    "has_xla_support",
    "has_native_dist_support",
    "has_hvd_support",
    "sync",
    "registered_computation_models",
    "one_rank_only",
    "new_group",
    "one_rank_first",
]

_model = _SerialModel()

_need_to_sync = True


def sync(temporary: bool = False) -> None:
    """Helper method to force this module to synchronize with current distributed context.
    This method should be used when distributed context is manually created or destroyed.

    Args:
        temporary: If True, distributed model synchronization is done every call of ``idist.get_*`` methods.
            This may have a negative performance impact.
    """
    global _model

    for comp_model_cls in registered_computation_models:
        if comp_model_cls == _SerialModel:
            continue
        model = comp_model_cls.create_from_context()
        if model is not None:
            _set_model(model, temporary=temporary)
            return

    _model = _SerialModel()


def device() -> torch.device:
    """Returns current device according to current distributed configuration.

    - `torch.device("cpu")` if no distributed configuration or torch native gloo distributed configuration
    - `torch.device("cuda:local_rank")` if torch native nccl or horovod distributed configuration
    - `torch.device("xla:index")` if XLA distributed configuration

    Returns:
        torch.device

    .. versionchanged:: 0.4.2
        Added Horovod distributed framework.
    """
    if _need_to_sync and isinstance(_model, _SerialModel):
        sync(temporary=True)

    return _model.device()


def backend() -> Optional[str]:
    """Returns computation model's backend.

    - `None` for no distributed configuration
    - "nccl" or "gloo" or "mpi" for native torch distributed configuration
    - "xla-tpu" for XLA distributed configuration
    - "horovod" for Horovod distributed framework

    Returns:
        str or None

    .. versionchanged:: 0.4.2
        Added Horovod distributed framework.
    """
    if _need_to_sync and isinstance(_model, _SerialModel):
        sync(temporary=True)

    return _model.backend()


def available_backends() -> Tuple[str, ...]:
    """Returns available backends."""
    out: Tuple[str, ...] = ()
    for m in registered_computation_models:
        out += m.available_backends
    return out


def model_name() -> str:
    """Returns distributed configuration name (given by ignite)

    - `serial` for no distributed configuration
    - `native-dist` for native torch distributed configuration
    - `xla-dist` for XLA distributed configuration
    - `horovod-dist` for Horovod distributed framework

    .. versionchanged:: 0.4.2
        `horovod-dist` will be returned for Horovod distributed framework.
    """
    if _need_to_sync and isinstance(_model, _SerialModel):
        sync(temporary=True)

    return _model.name


def get_world_size() -> int:
    """Returns world size of current distributed configuration. Returns 1 if no distributed configuration."""
    if _need_to_sync and isinstance(_model, _SerialModel):
        sync(temporary=True)

    return _model.get_world_size()


def get_rank() -> int:
    """Returns process rank within current distributed configuration. Returns 0 if no distributed configuration."""
    if _need_to_sync and isinstance(_model, _SerialModel):
        sync(temporary=True)

    return _model.get_rank()


def get_local_rank() -> int:
    """Returns local process rank within current distributed configuration.
    Returns 0 if no distributed configuration."""
    if _need_to_sync and isinstance(_model, _SerialModel):
        sync(temporary=True)

    return _model.get_local_rank()


def get_nproc_per_node() -> int:
    """Returns number of processes (or tasks) per node within current distributed configuration.
    Returns 1 if no distributed configuration.
    """
    if _need_to_sync and isinstance(_model, _SerialModel):
        sync(temporary=True)

    return _model.get_nproc_per_node()


def get_nnodes() -> int:
    """Returns number of nodes within current distributed configuration.
    Returns 1 if no distributed configuration.
    """
    if _need_to_sync and isinstance(_model, _SerialModel):
        sync(temporary=True)

    return _model.get_nnodes()


def get_node_rank() -> int:
    """Returns node rank within current distributed configuration.
    Returns 0 if no distributed configuration.
    """
    if _need_to_sync and isinstance(_model, _SerialModel):
        sync(temporary=True)

    return _model.get_node_rank()


def hostname() -> str:
    """Returns host name for current process within current distributed configuration."""
    return socket.gethostname()


def spawn(
    backend: str,
    fn: Callable,
    args: Tuple,
    kwargs_dict: Optional[Mapping] = None,
    nproc_per_node: int = 1,
    **kwargs: Any,
) -> None:
    """Spawns ``nproc_per_node`` processes that run ``fn`` with ``args``/``kwargs_dict`` and initialize
    distributed configuration defined by ``backend``.

    Args:
        backend: backend to use: `nccl`, `gloo`, `xla-tpu`, `horovod`
        fn: function to called as the entrypoint of the spawned process.
            This function must be defined at the top level of a module so it can be pickled and spawned.
            This is a requirement imposed by multiprocessing. The function is called as ``fn(i, *args, **kwargs_dict)``,
            where `i` is the process index and args is the passed through tuple of arguments.
        args: arguments passed to `fn`.
        kwargs_dict: kwargs passed to `fn`.
        nproc_per_node: number of processes to spawn on a single node. Default, 1.
        kwargs: acceptable kwargs according to provided backend:

            - | "nccl" or "gloo" : ``nnodes`` (default, 1), ``node_rank`` (default, 0), ``master_addr``
              | (default, "127.0.0.1"), ``master_port`` (default, 2222), ``init_method`` (default, "env://"),
              | `timeout` to `dist.init_process_group`_ function
              | and kwargs for `mp.start_processes`_ function.

            - | "xla-tpu" : ``nnodes`` (default, 1), ``node_rank`` (default, 0) and kwargs to `xmp.spawn`_ function.

            - | "horovod": ``hosts`` (default, None) and other kwargs to `hvd_run`_ function. Arguments ``nnodes=1``
              | and ``node_rank=0`` are tolerated and ignored, otherwise an exception is raised.

    Examples:
        1) Launch single node multi-GPU training using torch native distributed framework

        .. code-block:: python

            # >>> python main.py

            # main.py

            import ignite.distributed as idist

            def train_fn(local_rank, a, b, c, d=12):
                import torch.distributed as dist
                assert dist.is_available() and dist.is_initialized()
                assert dist.get_world_size() == 4

                device = idist.device()
                assert device == torch.device(f"cuda:{local_rank}")


            idist.spawn("nccl", train_fn, args=(a, b, c), kwargs_dict={"d": 23}, nproc_per_node=4)


        2) Launch multi-node multi-GPU training using torch native distributed framework

        .. code-block:: python

            # >>> (node 0): python main.py --node_rank=0 --nnodes=8 --master_addr=master --master_port=2222
            # >>> (node 1): python main.py --node_rank=1 --nnodes=8 --master_addr=master --master_port=2222
            # >>> ...
            # >>> (node 7): python main.py --node_rank=7 --nnodes=8 --master_addr=master --master_port=2222

            # main.py

            import torch
            import ignite.distributed as idist

            def train_fn(local_rank, nnodes, nproc_per_node):
                import torch.distributed as dist
                assert dist.is_available() and dist.is_initialized()
                assert dist.get_world_size() == nnodes * nproc_per_node

                device = idist.device()
                assert device == torch.device(f"cuda:{local_rank}")

            idist.spawn(
                "nccl",
                train_fn,
                args=(nnodes, nproc_per_node),
                nproc_per_node=nproc_per_node,
                nnodes=nnodes,
                node_rank=node_rank,
                master_addr=master_addr,
                master_port=master_port
            )

        3) Launch single node multi-TPU training (for example on Google Colab) using PyTorch/XLA

        .. code-block:: python

            # >>> python main.py

            # main.py

            import ignite.distributed as idist

            def train_fn(local_rank, a, b, c, d=12):
                import torch_xla.core.xla_model as xm
                assert xm.get_world_size() == 8

                device = idist.device()
                assert "xla" in device.type


            idist.spawn("xla-tpu", train_fn, args=(a, b, c), kwargs_dict={"d": 23}, nproc_per_node=8)

    .. _dist.init_process_group: https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
    .. _mp.start_processes: https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn.spawn
    .. _xmp.spawn: https://pytorch.org/xla/release/1.6/index.html#torch_xla.distributed.xla_multiprocessing.spawn
    .. _hvd_run: https://horovod.readthedocs.io/en/latest/api.html#module-horovod.run

    .. versionchanged:: 0.4.2
        ``backend`` now accepts `horovod` distributed framework.
    """
    _assert_backend(backend)

    if kwargs_dict is None:
        kwargs_dict = {}

    for comp_model_cls in registered_computation_models:
        if backend not in comp_model_cls.available_backends:
            continue
        comp_model_cls.spawn(
            fn, args=args, kwargs_dict=kwargs_dict, nproc_per_node=nproc_per_node, backend=backend, **kwargs
        )


def all_reduce(
    tensor: Union[torch.Tensor, float], op: str = "SUM", group: Optional[Union[Any, List[int]]] = None
) -> Union[torch.Tensor, float]:
    """Helper method to perform all reduce operation.

    Args:
        tensor: tensor or number to collect across participating processes.
        op: reduction operation, "SUM" by default. Possible values: "SUM", "PRODUCT", "MIN", "MAX", "AND", "OR".
            Horovod backend supports only "SUM", "AVERAGE", "ADASUM", "MIN", "MAX", "PRODUCT".
        group: list of integer or the process group for each backend. If None, the default process group will be used.

    Returns:
        torch.Tensor or number

    .. versionchanged:: 0.4.11
        added ``group``
    """
    if _need_to_sync and isinstance(_model, _SerialModel):
        sync(temporary=True)

    if isinstance(group, list) and all(isinstance(item, int) for item in group):
        group = _model.new_group(group)

    return _model.all_reduce(tensor, op, group=group)


def all_gather(
    tensor: Union[torch.Tensor, float, str, Any], group: Optional[Union[Any, List[int]]] = None
) -> Union[torch.Tensor, float, List[float], List[str], List[Any]]:
    """Helper method to perform all gather operation.

    Args:
        tensor: tensor or number or str to collect across participating processes. If tensor, it should have the
            same shape across processes.
        group: list of integer or the process group for each backend. If None, the default process group will be used.

    Returns:
        If input is a tensor, returns a torch.Tensor of shape ``(world_size * tensor.shape[0], tensor.shape[1], ...)``.
        If input is a number, a torch.Tensor of shape ``(world_size, )`` is returned and finally a list of strings
        is returned if input is a string. If current process does not belong to `group`, the very ``tensor`` is
        returned.

    .. versionchanged:: 0.4.11
        added ``group``
    """
    if _need_to_sync and isinstance(_model, _SerialModel):
        sync(temporary=True)

    if isinstance(group, list) and all(isinstance(item, int) for item in group):
        group = _model.new_group(group)

    return _model.all_gather(tensor, group=group)


def broadcast(
    tensor: Union[torch.Tensor, float, str, None], src: int = 0, safe_mode: bool = False
) -> Union[torch.Tensor, float, str]:
    """Helper method to perform broadcast operation.

    Args:
        tensor: tensor or number or str to broadcast to participating processes.
            Make sure to respect data type of torch tensor input for all processes, otherwise execution will crash.
            Can use None for non-source data with ``safe_mode=True``.
        src: source rank. Default, 0.
        safe_mode: if True, non source input data can be ``None`` or anything (will be discarded), otherwise data type
            of the input ``tensor`` should be respected for all processes. Please, keep in mind, this mode is working
            only for dense tensors as source input if a tensor is provided. It also leads to some additional
            collectives before the broadcast, making it slower than without using this mode. Default, False.

    Returns:
        torch.Tensor or string or number

    Examples:
        .. code-block:: python

            y = None
            if idist.get_rank() == 0:
                t1 = torch.rand(4, 5, 6, device=idist.device())
                s1 = "abc"
                x = 12.3456
                y = torch.rand(1, 2, 3, device=idist.device())
            else:
                t1 = torch.empty(4, 5, 6, device=idist.device())
                s1 = ""
                x = 0.0

            # Broadcast tensor t1 from rank 0 to all processes
            t1 = idist.broadcast(t1, src=0)
            assert isinstance(t1, torch.Tensor)

            # Broadcast string s1 from rank 0 to all processes
            s1 = idist.broadcast(s1, src=0)
            # >>> s1 = "abc"

            # Broadcast float number x from rank 0 to all processes
            x = idist.broadcast(x, src=0)
            # >>> x = 12.3456

            # Broadcast any of those types from rank 0,
            # but other ranks do not define the placeholder
            y = idist.broadcast(y, src=0, safe_mode=True)
            assert isinstance(y, torch.Tensor)

    .. versionadded:: 0.4.2

    .. versionchanged:: 0.4.5
        added ``safe_mode``
    """
    if _need_to_sync and isinstance(_model, _SerialModel):
        sync(temporary=True)

    return _model.broadcast(tensor, src=src, safe_mode=safe_mode)


def barrier() -> None:
    """Helper method to synchronize all processes."""
    if _need_to_sync and isinstance(_model, _SerialModel):
        sync(temporary=True)

    _model.barrier()


def new_group(ranks: List[int], **kwargs: Any) -> Any:
    """Helper method to make group for each backend from ranks.

    Args:
        ranks: subset of ranks to be grouped.
        kwargs: acceptable kwargs according to provided backend:

            - | "nccl" or "gloo" : ``backend (=None)``, ``pg_options (=None)``.

    Examples:
        Launch single node multi-GPU training with ``torchrun`` utility.

        .. code-block:: python

            import ignite.distributed as idist

            ranks = [0, 1]
            group = idist.new_group(ranks)

    .. versionadded:: 0.4.11
    """

    if _need_to_sync and isinstance(_model, _SerialModel):
        sync(temporary=True)

    return _model.new_group(ranks, **kwargs)


def set_local_rank(index: int) -> None:
    """Method to hint the local rank in case if torch native distributed context is created by user
    without using :meth:`~ignite.distributed.utils.initialize` or :meth:`~ignite.distributed.utils.spawn`.

    Args:
        index: local rank or current process index

    Examples:
        User set up torch native distributed process group

        .. code-block:: python

            import ignite.distributed as idist

            def run(local_rank, *args, **kwargs):

                idist.set_local_rank(local_rank)
                # ...
                dist.init_process_group(**dist_info)
                # ...
    """
    from ignite.distributed.comp_models.base import ComputationModel

    ComputationModel._ext_local_rank = index


def _set_model(model: Any, temporary: bool = False) -> None:
    global _model, _need_to_sync
    _model = model
    _need_to_sync = True
    if not isinstance(_model, _SerialModel) and not temporary:
        _need_to_sync = False


def _assert_backend(backend: str) -> None:
    backends = available_backends()
    if backend not in backends:
        raise ValueError(f"Backend should be one of '{backends}'")


def initialize(backend: str, **kwargs: Any) -> None:
    """Initializes distributed configuration according to provided ``backend``

    Args:
        backend: backend: `nccl`, `gloo`, `xla-tpu`, `horovod`.
        kwargs: acceptable kwargs according to provided backend:

            - | "nccl" or "gloo" : ``timeout(=timedelta(minutes=30))``, ``init_method(=None)``,
              | ``rank(=None)``, ``world_size(=None)``.
              | By default, ``init_method`` will be "env://". See more info about parameters: `torch_init`_.

            - | "horovod" : comm(=None), more info: `hvd_init`_.

    Examples:
        Launch single node multi-GPU training with ``torchrun`` utility.

        .. code-block:: python

            # >>> torchrun --nproc_per_node=4 main.py
            # main.py

            import ignite.distributed as idist

            def train_fn(local_rank, a, b, c):
                import torch.distributed as dist
                assert dist.is_available() and dist.is_initialized()
                assert dist.get_world_size() == 4

                device = idist.device()
                assert device == torch.device(f"cuda:{local_rank}")


            backend = "nccl"  # or "gloo" or "horovod" or "xla-tpu"
            idist.initialize(backend)
            # or for torch native distributed on Windows:
            # idist.initialize("nccl", init_method="file://tmp/shared")
            local_rank = idist.get_local_rank()
            train_fn(local_rank, a, b, c)
            idist.finalize()


    .. _torch_init: https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
    .. _hvd_init: https://horovod.readthedocs.io/en/latest/api.html#module-horovod.torch

    .. versionchanged:: 0.4.2
        ``backend`` now accepts `horovod` distributed framework.

    .. versionchanged:: 0.4.5
        ``kwargs`` now accepts ``init_method``, ``rank``, ``world_size`` for PyTorch native distributed backend.
    """
    if not (has_xla_support or has_native_dist_support or has_hvd_support):
        # nothing to do => serial model
        # maybe warn about this
        return

    _assert_backend(backend)

    for comp_model_cls in registered_computation_models:
        if backend not in comp_model_cls.available_backends:
            continue
        _set_model(comp_model_cls(backend, **kwargs))


def finalize() -> None:
    """Finalizes distributed configuration. For example, in case of native pytorch distributed configuration,
    it calls ``dist.destroy_process_group()``.
    """
    _model.finalize()
    _set_model(_SerialModel())


def show_config() -> None:
    """Helper method to display distributed configuration via ``logging``."""

    # setup parallel logger
    logger = setup_logger(__name__)

    logger.info(f"distributed configuration: {model_name()}")
    logger.info(f"backend: {backend()}")
    logger.info(f"device: {device().type}")
    logger.info(f"hostname: {hostname()}")
    logger.info(f"world size: {get_world_size()}")
    logger.info(f"rank: {get_rank()}")
    logger.info(f"local rank: {get_local_rank()}")
    logger.info(f"num processes per_node: {get_nproc_per_node()}")
    logger.info(f"num nodes: {get_nnodes()}")
    logger.info(f"node rank: {get_node_rank()}")


def one_rank_only(rank: int = 0, with_barrier: bool = False) -> Callable:
    """Decorator to filter handlers wrt a rank number

    Args:
        rank: rank number of the handler (default: 0).
        with_barrier: synchronisation with a barrier (default: False).

    Examples:
        .. code-block:: python

            engine = ...

            @engine.on(...)
            @one_rank_only() # means @one_rank_only(rank=0)
            def some_handler(_):
                ...

            @engine.on(...)
            @one_rank_only(rank=1)
            def some_handler(_):
                ...
    """

    def _one_rank_only(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Optional[Any]:
            ret = None
            if get_rank() == rank:
                ret = func(*args, **kwargs)
            if with_barrier:
                barrier()
            return ret

        return wrapper

    return _one_rank_only


@contextmanager
def one_rank_first(rank: int = 0, local: bool = False) -> Any:
    """Context manager that ensures a specific rank runs first before others in a distributed
    environment.

    Args:
        rank: rank of the process that should execute the code
            block inside the context manager first. Default, 0.
        local: flag to specify local rank or global rank.
            If True ``rank`` argument will define a local rank to run first.
            Default, False

    Examples:
        .. code-block:: python

            def download_dataset():
                ...

            with idist.one_rank_first():
                ds = download_dataset()

            dp = ds[0]

    .. versionadded:: 0.4.13
    """

    current_rank = get_local_rank() if local else get_rank()
    size = get_nproc_per_node() if local else get_world_size()

    if rank >= size or rank < 0:
        raise ValueError(f"rank should be between 0 and {size - 1}, but given {rank}")

    if current_rank != rank:
        barrier()

    yield

    if current_rank == rank:
        barrier()
