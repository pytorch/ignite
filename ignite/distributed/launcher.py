from typing import Any, Callable, Dict, Optional

from ignite.distributed import utils as idist
from ignite.utils import setup_logger

__all__ = [
    "Parallel",
]


class Parallel:
    """Distributed launcher context manager to simplify distributed configuration setup for multiple backends:

    - backends from native torch distributed configuration: "nccl", "gloo", "mpi" (if available)

    - XLA on TPUs via `pytorch/xla <https://github.com/pytorch/xla>`_ (if installed)

    - using `Horovod distributed framework <https://horovod.readthedocs.io>`_ (if installed)

    Namely, it can 1) spawn ``nproc_per_node`` child processes and initialize a processing group according to
    provided ``backend`` (useful for standalone scripts) or 2) only initialize a processing group given the ``backend``
    (useful with tools like `torch.distributed.launch`_, `horovodrun`_, etc).

    Examples:

        1) Single node or Multi-node, Multi-GPU training launched with `torch.distributed.launch`_ or `horovodrun`_
        tools

        Single node option with 4 GPUs

        .. code-block:: bash

            python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py
            # or if installed horovod
            horovodrun -np=4 python main.py

        Multi-node option : 2 nodes with 8 GPUs each

        .. code-block:: bash

            ## node 0
            python -m torch.distributed.launch --nnodes=2 --node_rank=0 --master_addr=master \
                --master_port=3344 --nproc_per_node=8 --use_env main.py

            # or if installed horovod
            horovodrun -np 16 -H hostname1:8,hostname2:8 python main.py

            ## node 1
            python -m torch.distributed.launch --nnodes=2 --node_rank=1 --master_addr=master \
                --master_port=3344 --nproc_per_node=8 --use_env main.py


        User code is the same for both options:

        .. code-block:: python

            # main.py

            import ignite.distributed as idist

            def training(local_rank, config, **kwargs):
                # ...
                print(idist.get_rank(), ": run with config:", config, "- backend=", idist.backend())
                # ...

            backend = "nccl"  # or "horovod" if package is installed

            with idist.Parallel(backend=backend) as parallel:
                parallel.run(training, config, a=1, b=2)


        2) Single node, Multi-GPU training launched with `python`

        .. code-block:: bash

            python main.py

        .. code-block:: python

            # main.py

            import ignite.distributed as idist

            def training(local_rank, config, **kwargs):
                # ...
                print(idist.get_rank(), ": run with config:", config, "- backend=", idist.backend())
                # ...

            backend = "nccl"  # or "horovod" if package is installed

            with idist.Parallel(backend=backend, nproc_per_node=4) as parallel:
                parallel.run(training, config, a=1, b=2)


        3) Single node, Multi-TPU training launched with `python`

        .. code-block:: bash

            python main.py

        .. code-block:: python

            # main.py

            import ignite.distributed as idist

            def training(local_rank, config, **kwargs):
                # ...
                print(idist.get_rank(), ": run with config:", config, "- backend=", idist.backend())
                # ...

            with idist.Parallel(backend="xla-tpu", nproc_per_node=8) as parallel:
                parallel.run(training, config, a=1, b=2)


        4) Multi-node, Multi-GPU training launched with `python`. For example, 2 nodes with 8 GPUs:

        Using torch native distributed framework:

        .. code-block:: bash

            # node 0
            python main.py --node_rank=0

            # node 1
            python main.py --node_rank=1


        .. code-block:: python

            # main.py

            import ignite.distributed as idist

            def training(local_rank, config, **kwargs):
                # ...
                print(idist.get_rank(), ": run with config:", config, "- backend=", idist.backend())
                # ...

            dist_config = {
                "nproc_per_node": 8,
                "nnodes": 2,
                "node_rank": args.node_rank,
                "master_addr": "master",
                "master_port": 15000
            }

            with idist.Parallel(backend="nccl", **dist_config) as parallel:
                parallel.run(training, config, a=1, b=2)

    .. _torch.distributed.launch: https://pytorch.org/docs/stable/distributed.html#launch-utility
    .. _horovodrun: https://horovod.readthedocs.io/en/latest/api.html#module-horovod.run

    Args:
        backend (str, optional): backend to use: `nccl`, `gloo`, `xla-tpu`, `horovod`. If None, no distributed
            configuration.
        nproc_per_node (int, optional): optional argument, number of processes per
            node to specify. If not None, :meth:`~ignite.distributed.Parallel.run` will spawn ``nproc_per_node``
            processes that run input function with its arguments.
        nnodes (int, optional): optional argument, number of nodes participating in distributed configuration.
            If not None, :meth:`~ignite.distributed.Parallel.run` will spawn ``nproc_per_node``
            processes that run input function with its arguments. Total world size is `nproc_per_node * nnodes`.
            This option is only supported by native torch distributed module. For other modules, please setup
            ``spawn_kwargs`` with backend specific arguments.
        node_rank (int, optional): optional argument, current machine index. Mandatory argument if ``nnodes`` is
            specified and larger than one.
            This option is only supported by native torch distributed module. For other modules, please setup
            ``spawn_kwargs`` with backend specific arguments.
        master_addr (str, optional): optional argument, master node TCP/IP address for torch native backends
            (`nccl`, `gloo`). Mandatory argument if ``nnodes`` is specified and larger than one.
        master_port (int, optional): optional argument, master node port for torch native backends
            (`nccl`, `gloo`). Mandatory argument if ``master_addr`` is specified.
        **spawn_kwargs: kwargs to ``idist.spawn`` function.

    .. versionchanged:: 0.4.2
        ``backend`` now accepts `horovod` distributed framework.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        nproc_per_node: Optional[int] = None,
        nnodes: Optional[int] = None,
        node_rank: Optional[int] = None,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        **spawn_kwargs: Any,
    ) -> None:
        if backend is not None:
            if backend not in idist.available_backends():
                raise ValueError(f"Unknown backend '{backend}'. Available backends: {idist.available_backends()}")
        else:
            arg_names = ["nproc_per_node", "nnodes", "node_rank", "master_addr", "master_port"]
            arg_values = [nproc_per_node, nnodes, node_rank, master_addr, master_port]
            for name, value in zip(arg_names, arg_values):
                if value is not None:
                    raise ValueError(f"If backend is None, argument '{name}' should be also None, but given {value}")

        self.backend = backend
        self._spawn_params = None
        self.logger = setup_logger(__name__ + "." + self.__class__.__name__, distributed_rank=0)
        # distributed_rank=0 <=> explicit rank 0, avoid call idist. Critical for TPU on Colab, avoid context is setup

        if self.backend is not None:
            if nproc_per_node is not None:
                self._spawn_params = self._setup_spawn_params(
                    nproc_per_node, nnodes, node_rank, master_addr, master_port, **spawn_kwargs
                )

        if self._spawn_params is not None:
            self.logger.info(f"Initialized distributed launcher with backend: '{self.backend}'")
            msg = "\n\t".join([f"{k}: {v}" for k, v in self._spawn_params.items() if v is not None])
            self.logger.info(f"- Parameters to spawn processes: \n\t{msg}")

    @staticmethod
    def _setup_spawn_params(
        nproc_per_node: int,
        nnodes: Optional[int] = None,
        node_rank: Optional[int] = None,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        **spawn_kwargs: Any,
    ) -> Dict:
        if nproc_per_node < 1:
            raise ValueError(f"Argument nproc_per_node should positive, but given {nproc_per_node}")
        if nnodes is None:
            nnodes = 1
        if nnodes < 1:
            raise ValueError(f"Argument nnodes should positive, but given {nnodes}")
        if node_rank is None:
            if nnodes > 1:
                raise ValueError("If number of nodes larger than one, arguments node_rank should be given")
            node_rank = 0
        if node_rank >= nnodes or node_rank < 0:
            raise ValueError(f"Argument node_rank should be between 0 and {nnodes - 1}, but given {node_rank}")
        if nnodes > 1 and (master_addr is None or master_port is None):
            raise ValueError(
                "If number of nodes larger than one, arguments master_addr and master_port "
                f"should be specified, but given master_addr={master_addr} and master_port={master_port}"
            )
        params = {
            "nproc_per_node": nproc_per_node,
            "nnodes": nnodes,
            "node_rank": node_rank,
            "master_addr": master_addr,
            "master_port": master_port,
        }
        params.update(spawn_kwargs)
        return {k: v for k, v in params.items() if v is not None}

    def run(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        """Execute ``func`` with provided arguments in distributed context.

        Example

        .. code-block:: python

            def training(local_rank, config, **kwargs):
                # ...
                print(idist.get_rank(), ": run with config:", config, "- backend=", idist.backend())
                # ...

            with idist.Parallel(backend=backend) as parallel:
                parallel.run(training, config, a=1, b=2)

        Args:
            func (Callable): function to execute. First argument of the function should be `local_rank` - local process
                index.
            *args: positional arguments of ``func`` (without `local_rank`).
            **kwargs: keyword arguments of ``func``.

        """
        if self._spawn_params is not None and self.backend is not None:
            self.logger.info(f"Spawn function '{func}' in {self._spawn_params['nproc_per_node']} processes")
            idist.spawn(self.backend, func, args=args, kwargs_dict=kwargs, **self._spawn_params)
        else:
            self.logger.info(f"- Run '{func}' in {idist.get_world_size()} processes")
            local_rank = idist.get_local_rank()
            func(local_rank, *args, **kwargs)

        self.logger.info("End of run")

    def __enter__(self) -> "Parallel":
        if (self.backend is not None) and self._spawn_params is None:
            idist.initialize(self.backend)
            self.logger = setup_logger(__name__ + "." + self.__class__.__name__)
            self.logger.info(f"Initialized processing group with backend: '{self.backend}'")

        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        if (self.backend is not None) and self._spawn_params is None:
            self.logger.info(f"Finalized processing group with backend: '{self.backend}'")
            idist.finalize()
