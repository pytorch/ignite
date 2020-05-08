from typing import Optional, Union

import torch

from ignite.distributed.comp_models.base import ComputationModel

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

    has_xla_support = True
except ImportError:
    has_xla_support = False


class _XlaDistModel(ComputationModel):
    """Private class for PyTorch XLA basic distributed computation model.

    Supported XLA devices:

    - CPU
    - TPU

    """

    name = "xla-dist"

    available_backends = tuple(["xla-tpu",])

    @staticmethod
    def create_from_context() -> Optional["_XlaDistModel"]:
        if not has_xla_support:
            return None
        raise NotImplementedError("")

    @staticmethod
    def create_from_backend(backend: str = "xla-tpu", **kwargs) -> "_XlaDistModel":
        if not has_xla_support:
            raise RuntimeError("Torch xla package is not installed.")
        return _XlaDistModel(backend=backend, **kwargs)

    def __init__(self, backend=None, **kwargs):
        """This is a private method. Please, use `create_from_backend` or `create_from_context`
        """

        if backend is not None:
            self._create_from_backend(backend, **kwargs)
        else:
            self._init_from_context()

    def _create_from_backend(self, backend, **kwargs):
        xm.rendezvous("init")

        self._backend = backend
        self._ntasks_per_node = self._compute_ntasks_per_node()
        self._nnodes = self.get_world_size() // self.get_ntasks_per_node()
        self._node = self.get_rank() // self._ntasks_per_node

    def _init_from_context(self):
        raise NotImplementedError("")

    def _compute_ntasks_per_node(self):
        tensor = torch.tensor([self.get_local_rank() + 1.0], dtype=torch.float).to(self.device())
        xm.all_reduce("max", [tensor,])
        return int(tensor.item())

    def get_local_rank(self) -> int:
        return xm.get_local_ordinal()

    def get_rank(self) -> int:
        return xm.get_ordinal()

    def get_world_size(self) -> int:
        return xm.xrt_world_size()

    def get_ntasks_per_node(self) -> int:
        return self._ntasks_per_node

    def get_num_nodes(self) -> int:
        return self._nnodes

    def get_node_rank(self) -> int:
        return self._node

    def device(self) -> Union[torch.device, str]:
        return xm.xla_device()

    def backend(self) -> Optional[str]:
        return self._backend

    def finalize(self):
        pass

    @staticmethod
    def _dist_worker_task_fn(local_rank, backend, fn, args):
        from ignite.distributed.utils import _set_model

        model = _XlaDistModel.create_from_backend(backend)
        _set_model(model)
        fn(local_rank, *args)
        model.finalize()

    @staticmethod
    def spawn(fn, args, num_procs_per_node, num_nodes=1, node_rank=0, backend="xla-tpu", **kwargs):
        if not has_xla_support:
            raise RuntimeError("Torch xla package is not installed.")

        import os

        spawn_kwargs = {}
        if "COLAB_TPU_ADDR" in os.environ:
            spawn_kwargs["start_method"] = "fork"

        xmp.spawn(
            _XlaDistModel._dist_worker_task_fn, args=(backend, fn, args), nprocs=num_procs_per_node, **spawn_kwargs
        )
