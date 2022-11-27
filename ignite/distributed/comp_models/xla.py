from typing import Any, Callable, cast, List, Mapping, Optional, Tuple

import torch

from ignite.distributed.comp_models.base import ComputationModel

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

    has_xla_support = True
except ImportError:
    has_xla_support = False


if has_xla_support:

    XLA_TPU = "xla-tpu"

    class _XlaDistModel(ComputationModel):
        """Private class for PyTorch XLA basic distributed computation model.
        It handles single/multi-device computation model.

        Supported XLA devices:

        - CPU
        - TPU

        """

        name = "xla-dist"

        available_backends = (XLA_TPU,)

        @staticmethod
        def create_from_context() -> Optional["_XlaDistModel"]:
            return _XlaDistModel()

        @staticmethod
        def create_from_backend(backend: str = XLA_TPU, **kwargs: Any) -> "_XlaDistModel":
            if backend not in _XlaDistModel.available_backends:
                raise ValueError(f"Backend should be one of '{_XlaDistModel.available_backends}'")

            return _XlaDistModel(backend=backend, **kwargs)

        def __init__(self, backend: Optional[str] = None, **kwargs: Any):
            """This is a private method. Please, use `create_from_backend` or `create_from_context`"""
            super(_XlaDistModel, self).__init__()
            if backend is not None:
                self._create_from_backend(backend, **kwargs)
            else:
                self._init_from_context()

        def _create_from_backend(self, backend: str, **kwargs: Any) -> None:
            xm.rendezvous("init")

            self._backend: str = backend
            self._setup_attrs()

        def _init_from_context(self) -> None:
            self._backend = XLA_TPU
            self._setup_attrs()

        def _compute_nproc_per_node(self) -> int:
            tensor = torch.tensor([self.get_local_rank() + 1.0], dtype=torch.float).to(self.device())
            xm.all_reduce("max", [tensor])
            return int(tensor.item())

        def get_local_rank(self) -> int:
            return xm.get_local_ordinal()

        def get_rank(self) -> int:
            return xm.get_ordinal()

        def get_world_size(self) -> int:
            return xm.xrt_world_size()

        def get_nproc_per_node(self) -> int:
            return cast(int, self._nproc_per_node)

        def get_nnodes(self) -> int:
            return cast(int, self._nnodes)

        def get_node_rank(self) -> int:
            return cast(int, self._node)

        def device(self) -> torch.device:
            dev = torch_xla._XLAC._xla_get_default_device()
            return torch.device(dev)

        def backend(self) -> str:
            return self._backend

        def finalize(self) -> None:
            pass

        @staticmethod
        def _dist_worker_task_fn(
            local_rank: int, backend: str, fn: Callable, args: Tuple, kwargs_dict: Mapping
        ) -> None:
            from ignite.distributed.utils import _set_model, finalize

            model = _XlaDistModel.create_from_backend(backend)
            _set_model(model)
            fn(local_rank, *args, **kwargs_dict)
            finalize()

        @staticmethod
        def spawn(
            fn: Callable,
            args: Tuple,
            kwargs_dict: Optional[Mapping] = None,
            nproc_per_node: int = 1,
            nnodes: int = 1,
            node_rank: int = 0,
            backend: str = XLA_TPU,
            **kwargs: Any,
        ) -> None:
            if "start_method" not in kwargs:
                kwargs["start_method"] = "fork"

            xmp.spawn(
                _XlaDistModel._dist_worker_task_fn,
                args=(backend, fn, args, kwargs_dict),
                nprocs=nproc_per_node,
                **kwargs,
            )

        _collective_op_dtype = torch.float32
        _reduce_op_map = {
            "SUM": "sum",
            "PRODUCT": "mul",
            "MIN": "min",
            "MAX": "max",
            "AND": "and",
            "OR": "or",
        }

        def _do_all_reduce(self, tensor: torch.Tensor, op: str = "SUM", group: Optional[Any] = None) -> torch.Tensor:
            if group is not None and not self._check_group_type(group):
                raise ValueError("Argument group should be list of int")
            op = self._reduce_op_map[op]
            xm.all_reduce(op, [tensor], groups=group)
            return tensor

        def _do_all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None) -> torch.Tensor:
            # from https://github.com/jysohn23/xla/blob/model-parallel-colab/Gather_Scatter_Broadcast_PyTorch_XLA.ipynb

            if group is not None and (not isinstance(group, list) or not all(isinstance(item, int) for item in group)):
                raise ValueError("Argument group should be list of int")

            group_size = self.get_world_size()
            output = torch.zeros((group_size,) + tensor.shape, dtype=tensor.dtype, device=tensor.device)
            output[self.get_rank() % group_size] = tensor
            xm.all_reduce("sum", [output], groups=group)
            return output.reshape(-1, *output.shape[2:])

        def _do_new_group(self, ranks: List[int], **kwargs: Any) -> Any:
            return [ranks]

        def _do_broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
            # from https://github.com/jysohn23/xla/blob/model-parallel-colab/Gather_Scatter_Broadcast_PyTorch_XLA.ipynb
            if src != self.get_rank():
                tensor.fill_(0.0)
            xm.all_reduce("sum", [tensor])
            return tensor

        def barrier(self) -> None:
            xm.rendezvous("barrier")

        def _check_group_type(self, group: Optional[Any]) -> bool:
            if isinstance(group, list) and all(isinstance(item, int) for item in group):
                return True
            return False
