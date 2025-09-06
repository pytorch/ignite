import os
from distutils.version import LooseVersion
from typing import Any, Callable, Mapping, Optional, Tuple, cast

import torch
import torch.multiprocessing as mp
import torch.nn.parallel

from ignite.distributed.comp_models.native import _NativeDistModel

try:
    import torch.distributed as dist
    import torch_ccl

    has_ccl_support = True
except ImportError:
    has_ccl_support = False


if has_ccl_support:

    TORCH_CCL = "torch_ccl"
    CCL_BACKEND = "ccl"

    class _CCLDistModel(_NativeDistModel):
        """Private class for Torch basic distributed computation model.
        It handles single/multi-device computation model.

        """

        name = "torch-ccl"

        available_backends = (TORCH_CCL,)

        def __init__(self, backend: str, **kwargs: Any):
            """This is a private method. Please, use `create_from_backend` or `create_from_context`
            """
            super(_CCLDistModel, self).__init__()
            self._create_from_backend(backend, **kwargs)

        def _create_from_backend(
            self,
            backend: str,
            timeout: Optional[int] = None,
            init_method: Optional[str] = None,
            world_size: Optional[int] = None,
            rank: Optional[int] = None,
            **kwargs: Any,
        ) -> None:
            self._backend = backend
            self.setup_env_vars(rank, world_size)

            init_pg_kwargs = {}
            if timeout is not None:
                init_pg_kwargs["timeout"] = timeout

            if init_method is None:
                init_method = "env://"

            if "env" not in init_method:
                init_pg_kwargs["world_size"] = int(os.environ["WORLD_SIZE"])
                init_pg_kwargs["rank"] = int(os.environ["RANK"])
            self._init_method = init_method

            dist.init_process_group(backend, init_method=init_method, **init_pg_kwargs)

            # https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
            dist.barrier()
            self._setup_attrs()

        @staticmethod
        def _dist_worker_task_fn(
            local_rank: int,
            backend: str,
            fn: Callable,
            args: Tuple,
            kw_dict: Mapping,
            world_size: int,
            nprocs_per_node: int,
            node_rank: int,
            master_addr: Optional[str],
            master_port: Optional[str],
            init_method: str,
            kw: Any,
        ) -> None:
            from ignite.distributed.utils import _set_model, finalize

            copy_env_vars = os.environ.copy()

            rank = node_rank * nprocs_per_node + local_rank
            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)

            arg_world_size = world_size  # type: Optional[int]
            arg_rank = rank  # type: Optional[int]

            if init_method == "env://":
                os.environ["MASTER_ADDR"] = str(master_addr)
                os.environ["MASTER_PORT"] = str(master_port)

            model = _NativeDistModel.create_from_backend(
                backend, init_method=init_method, world_size=arg_world_size, rank=arg_rank, **kw
            )

            _set_model(model)

            fn(local_rank, *args, **kw_dict)
            finalize()

            os.environ.clear()
            os.environ.update(copy_env_vars)

        @staticmethod
        def spawn(  # type: ignore[override]
            fn: Callable,
            args: Tuple,
            kwargs_dict: Optional[Mapping] = None,
            nproc_per_node: int = 1,
            nnodes: int = 1,
            node_rank: int = 0,
            backend: str = CCL_BACKEND,
            init_method: Optional[str] = None,
            **kwargs: Any,
        ) -> None:
            world_size = nnodes * nproc_per_node

            spawn_kwargs = {
                "join": kwargs.get("join", True),
                "daemon": kwargs.get("daemon", False),
            }

            start_processes = mp.spawn
            # start_method and start_processes in pytorch >= 1.5
            if LooseVersion(torch.__version__) >= LooseVersion("1.5.0"):
                spawn_kwargs["start_method"] = kwargs.get("start_method", "spawn")
                start_processes = mp.start_processes

            if init_method in [None, "env://"]:
                init_method = "env://"
                master_addr = "127.0.0.1"
                master_port = 2222

            start_processes(
                _CCLDistModel._dist_worker_task_fn,
                nprocs=nproc_per_node,
                args=(
                    backend,
                    fn,
                    args,
                    kwargs_dict,
                    world_size,
                    nproc_per_node,
                    node_rank,
                    master_addr,
                    master_port,
                    init_method,
                    kwargs,
                ),
                **spawn_kwargs,
            )

        _reduce_op_map = {
            "SUM": dist.ReduceOp.SUM,
            "PRODUCT": dist.ReduceOp.PRODUCT,
            "MIN": dist.ReduceOp.MIN,
            "MAX": dist.ReduceOp.MAX,
            "AND": dist.ReduceOp.BAND,
            "OR": dist.ReduceOp.BOR,
        }
