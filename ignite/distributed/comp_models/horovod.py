import os
import warnings
from typing import Any, Callable, Mapping, Optional, Tuple, cast

import torch

from ignite.distributed.comp_models.base import ComputationModel

try:
    import horovod.torch as hvd

    try:
        # old API
        from horovod.run.runner import run as hvd_mp_spawn
    except ImportError:
        # new API: https://github.com/horovod/horovod/pull/2099
        from horovod import run as hvd_mp_spawn

    has_hvd_support = True
except ImportError:
    has_hvd_support = False


if has_hvd_support:

    HOROVOD = "horovod"

    class _HorovodDistModel(ComputationModel):
        """Private class for `Horovod <https://horovod.readthedocs.io/en/stable/>`_ distributed computation model.
        """

        name = "horovod-dist"

        available_backends = (HOROVOD,)

        @staticmethod
        def _get_hvd_rank() -> int:
            try:
                rank = hvd.rank()
            except ValueError as e:
                rank = -1
            return rank

        @staticmethod
        def create_from_context() -> Optional["_HorovodDistModel"]:
            rank = _HorovodDistModel._get_hvd_rank()
            if not (has_hvd_support and rank > -1):
                return None
            return _HorovodDistModel()

        @staticmethod
        def create_from_backend(backend: str, **kwargs: Any) -> "_HorovodDistModel":
            if backend not in _HorovodDistModel.available_backends:
                raise ValueError("Backend should be one of '{}'".format(_HorovodDistModel.available_backends))

            rank = _HorovodDistModel._get_hvd_rank()
            if has_hvd_support and rank > -1:
                raise RuntimeError("Can not re-initialize Horovod if it is already initialized")
            return _HorovodDistModel(do_init=True, **kwargs)

        def __init__(self, do_init: bool = False, **kwargs: Any) -> None:
            """This is a private method. Please, use `create_from_backend` or `create_from_context`
            """
            super(_HorovodDistModel, self).__init__()
            self._backend = HOROVOD  # type: str
            if do_init:
                comm = kwargs.get("comm", None)
                hvd.init(comm=comm)

            self._local_rank = hvd.local_rank()

            if do_init and torch.cuda.is_available():
                torch.cuda.set_device(self._local_rank)

            self._setup_attrs()

        def _compute_nproc_per_node(self) -> int:
            return hvd.local_size()

        def get_local_rank(self) -> int:
            return self._local_rank

        def get_rank(self) -> int:
            return hvd.rank()

        def get_world_size(self) -> int:
            return hvd.size()

        def get_nproc_per_node(self) -> int:
            return cast(int, self._nproc_per_node)

        def get_nnodes(self) -> int:
            return cast(int, self._nnodes)

        def get_node_rank(self) -> int:
            return cast(int, self._node)

        def device(self) -> torch.device:
            if torch.cuda.is_available():
                index = torch.cuda.current_device()
                if index < self.get_local_rank():
                    warnings.warn(
                        "Current device index is less than current local rank. "
                        "Please, make sure to call torch.cuda.set_device(local_rank)."
                    )
                return torch.device("cuda:{}".format(index))
            return torch.device("cpu")

        def backend(self) -> str:
            return self._backend

        def finalize(self) -> None:
            hvd.shutdown()

        @staticmethod
        def _dist_worker_task_fn(backend: str, fn: Callable, args: Tuple, kwargs_dict: Mapping) -> None:
            from ignite.distributed.utils import _set_model, finalize

            model = _HorovodDistModel.create_from_backend(backend)
            _set_model(model)
            fn(model.get_local_rank(), *args, **kwargs_dict)
            finalize()

        @staticmethod
        def spawn(  # type: ignore[override]
            fn: Callable,
            args: Tuple,
            kwargs_dict: Optional[Mapping] = None,
            nproc_per_node: int = 1,
            hosts: Optional[str] = None,
            backend: str = HOROVOD,
            **kwargs: Any
        ) -> None:
            c1 = "nnodes" in kwargs and kwargs["nnodes"] > 1
            c2 = "node_rank" in kwargs and kwargs["node_rank"] > 0
            if c1 or c2:
                raise RuntimeError(
                    "For multi-node configuration, please set 'hosts' argument instead according to horovod.run API."
                )
            if "nnodes" in kwargs:
                # Remove 'nnodes=1' as it is an unexpected keyword argument for horovod.run
                del kwargs["nnodes"]
            if "node_rank" in kwargs:
                # Remove 'node_rank=0' as it is an unexpected keyword argument for horovod.run
                del kwargs["node_rank"]

            hvd_mp_spawn(
                _HorovodDistModel._dist_worker_task_fn,
                args=(HOROVOD, fn, args, kwargs_dict),
                np=nproc_per_node,
                hosts=hosts,
                **kwargs,
            )

        _reduce_op_map = {
            "SUM": hvd.mpi_ops.Sum,
            "AVERAGE": hvd.mpi_ops.Average,
            "ADASUM": hvd.mpi_ops.Adasum,
        }

        def _do_all_reduce(self, tensor: torch.Tensor, op: str = "SUM") -> torch.Tensor:
            if op not in self._reduce_op_map:
                raise ValueError("Unsupported reduction operation: '{}'".format(op))
            op = self._reduce_op_map[op]
            return hvd.allreduce(tensor, op=op)

        def _do_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
            if tensor.ndimension() == 0:
                tensor = tensor.unsqueeze(0)
            return hvd.allgather(tensor)

        def _do_broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
            return hvd.broadcast(tensor, root_rank=src)

        def barrier(self) -> None:
            # https://github.com/horovod/horovod/issues/159#issuecomment-424834603
            # hvd.allreduce(torch.tensor(0, device=self.device()), name="barrier")
            hvd.allreduce(torch.tensor(0, device="cpu"), name="barrier")
