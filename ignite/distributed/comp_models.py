from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import os
import subprocess

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ignite.distributed.utils import has_xla_support


class ComputationModel(metaclass=ABCMeta):

    # vfdev: I do not understand the purpose of the method and when to use it
    # @abstractmethod
    # def is_initialized(self) -> bool:
    #     pass

    @abstractmethod
    def get_local_rank(self) -> int:
        pass

    @abstractmethod
    def get_rank(self) -> int:
        pass

    @abstractmethod
    def get_world_size(self) -> int:
        pass

    @abstractmethod
    def get_ntasks_per_node(self) -> int:
        pass

    @abstractmethod
    def get_num_nodes(self) -> int:
        pass

    @abstractmethod
    def get_node_rank(self) -> int:
        pass

    @abstractmethod
    def is_distributed(self) -> bool:
        pass

    @abstractmethod
    def device(self) -> Union[torch.device, str]:
        pass

    @abstractmethod
    def backend(self) -> Optional[str]:
        pass

    @abstractmethod
    def finalize(self):
        pass

    @staticmethod
    @abstractmethod
    def create_from_context() -> Optional["ComputationModel"]:
        pass

    @staticmethod
    @abstractmethod
    def create_from_backend(backend: str, **kwargs) -> "ComputationModel":
        pass

    @staticmethod
    @abstractmethod
    def spawn(*args, **kwargs):
        pass


class _SerialModel(ComputationModel):

    name = "serial"
    available_backends = tuple()

    # def is_initialized(self) -> bool:
    #     return True

    def get_local_rank(self) -> int:
        return 0

    def get_rank(self) -> int:
        return 0

    def get_world_size(self) -> int:
        return 1

    def get_ntasks_per_node(self) -> int:
        return 1

    def get_num_nodes(self) -> int:
        return 1

    def get_node_rank(self) -> int:
        return 0

    def is_distributed(self) -> bool:
        return False

    def device(self) -> Union[torch.device, str]:
        return "cpu"

    def backend(self) -> Optional[str]:
        return None

    def finalize(self):
        pass

    @staticmethod
    def create_from_context() -> Optional["_SerialModel"]:
        return _SerialModel()

    @staticmethod
    def create_from_backend(backend: str, **kwargs) -> "_SerialModel":
        return _SerialModel()

    @staticmethod
    def spawn(*args, **kwargs):
        raise NotImplementedError("Serial computation model does not implement spawn method")


class _DistModel(ComputationModel):
    """PyTorch native distributed computation model.

    Supported `backends <https://pytorch.org/docs/stable/distributed.html#backends>`_:

    - NCCL
    - GLOO
    - MPI

    In this implementation we assume the following mapping between backend and devices:

    - NCCL <-> GPU
    - GLOO <-> CPU
    - MPI  <-> CPU

    """

    name = "native-dist"
    available_backends = tuple(
        name
        for name in [dist.Backend.NCCL, dist.Backend.GLOO, dist.Backend.MPI]
        if getattr(dist, "is_{}_available".format(name))
    )

    @staticmethod
    def create_from_context() -> Optional["_DistModel"]:
        if not (dist.is_available() and dist.is_initialized()):
            return None
        return _DistModel()

    @staticmethod
    def create_from_backend(backend: str, **kwargs) -> "_DistModel":
        if dist.is_available() and dist.is_initialized():
            raise RuntimeError("Can not create new distributed process group if default one is already initialized")
        return _DistModel(backend=backend, **kwargs)

    def __init__(self, backend=None, timeout=None, **kwargs):
        """This is a private method. Please, use `create_from_backend` or `create_from_context`
        """
        if backend is not None:
            self._create_from_backend(backend, timeout=timeout, **kwargs)
        else:
            self._init_from_context()

    def _create_from_backend(self, backend, timeout=None, **kwargs):
        self.setup_env_vars()
        self._master_port = int(os.environ["MASTER_PORT"])
        self._master_addr = os.environ["MASTER_ADDR"]
        self._local_rank = int(os.environ["LOCAL_RANK"])

        init_pg_kwargs = {}
        if timeout is not None:
            init_pg_kwargs["timeout"] = timeout

        dist.init_process_group(backend, init_method="env://", **init_pg_kwargs)
        # https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
        dist.barrier()

        if backend == "nccl":
            torch.cuda.device(self._local_rank)

        self._ntasks_per_node = self._compute_ntasks_per_node()
        self._nnodes = self.get_world_size() // self.get_ntasks_per_node()
        self._node = self.get_rank() // self._ntasks_per_node
        # self._is_initialized = True

    def _init_from_context(self):
        self._master_port = None
        self._master_addr = None

        # THIS COULD HELP TO GET master addr/port if TCPStore is used.
        # HOWEVER, user can use FileStore or any other store.
        # try:
        #     store = dist.distributed_c10d._get_default_store()
        #     if isinstance(store, torch.distributed.TCPStore):
        #         self._master_port = None
        #         self._master_addr = None
        # except AttributeError:
        #     pass

        self._local_rank = 0  # self.get_rank() % self._ntasks_per_node
        self._ntasks_per_node = self._compute_ntasks_per_node()
        self._nnodes = self.get_world_size() // self._ntasks_per_node
        self._node = self.get_rank() // self._ntasks_per_node

    def _compute_ntasks_per_node(self):
        tensor = torch.tensor([self.get_local_rank() + 1]).to(self.device())
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor.item()

    def setup_env_vars(self):
        if "SLURM_JOB_ID" in os.environ:
            self._setup_env_in_slurm()
            return

        # check if all necessary env vars are set
        # if partially defined raise an error
        necessary_env_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
        all_env_vars_defined = [k in os.environ for k in necessary_env_vars]
        if any(all_env_vars_defined) and not all(all_env_vars_defined):
            raise RuntimeError(
                "PyTorch distributed configuration should define env variables '{}'".format(necessary_env_vars)
            )

        os.environ["RANK"] = os.environ.get("RANK", "0")
        os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "15000")

    def _setup_env_in_slurm(self):
        for k in ["SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS", "SLURM_JOB_NODELIST"]:
            if k not in os.environ:
                raise RuntimeError("SLURM distributed configuration is missing '{}' in env variables".format(k))

        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        # port should be the same over all process
        slurm_port = os.environ["SLURM_JOB_ID"]
        slurm_port = slurm_port[-4:]
        os.environ["MASTER_PORT"] = str(int(slurm_port) + 15000)
        # master address is the first hostname of nodes list
        hostnames = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]])
        os.environ["MASTER_ADDR"] = hostnames.split()[0].decode("utf-8")

    # def is_initialized(self) -> bool:
    #     return self._is_initialized

    def get_local_rank(self) -> int:
        return self._local_rank

    def get_rank(self) -> int:
        return dist.get_rank()

    def get_world_size(self) -> int:
        return dist.get_world_size()

    def get_ntasks_per_node(self) -> int:
        return self._ntasks_per_node

    def get_num_nodes(self) -> int:
        return self._nnodes

    def get_node_rank(self) -> int:
        return self._node

    def is_distributed(self) -> bool:
        return dist.is_available() and dist.is_initialized() and self.get_world_size() > 1

    def device(self) -> Union[torch.device, str]:
        if self.backend() == "nccl":
            return "cuda:{}".format(torch.cuda.current_device())
        return "cpu"

    def backend(self) -> Optional[str]:
        return dist.get_backend()

    def finalize(self):
        dist.destroy_process_group()

    @staticmethod
    def _dist_worker_task_fn(
        local_rank, backend, fn, world_size, num_workers_per_machine, machine_rank, master_addr, master_port, args
    ):
        from ignite.distributed.utils import _set_model

        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(machine_rank * num_workers_per_machine + local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)

        model = _DistModel.create_from_backend(backend)
        _set_model(model)
        fn(local_rank, *args)
        model.finalize()

    @staticmethod
    def spawn(
        backend,
        fn,
        args,
        num_workers_per_machine,
        num_machines=1,
        machine_rank=0,
        master_addr="0.0.0.0",
        master_port=2222,
        **kwargs
    ):
        world_size = num_machines * num_workers_per_machine
        mp.spawn(
            _DistModel._dist_worker_task_fn,
            nprocs=num_workers_per_machine,
            args=(backend, fn, world_size, num_workers_per_machine, machine_rank, master_addr, master_port, args),
            daemon=False,
        )


class _XlaDistModel(ComputationModel):

    name = "xla-dist"

    available_backends = tuple("xla-tpu",)

    def __init__(self, *args, **kwargs):

        raise NotImplementedError("Not implemented")

        if not has_xla_support:
            raise RuntimeError("Torch xla package is not installed.")

        import torch_xla.core.xla_model as xm

        self._xm = xm

    # def is_initialized(self) -> bool:
    #     return True

    def get_local_rank(self) -> int:
        self._xm.get_ordinal()

    def get_rank(self) -> int:
        return self._xm.get_ordinal()

    def get_world_size(self) -> int:
        return self._xm.xrt_world_size()

    def get_ntasks_per_node(self) -> int:
        raise NotImplementedError("not yet implemented")

    def get_num_nodes(self) -> int:
        raise NotImplementedError("not yet implemented")

    def get_node_rank(self) -> int:
        raise NotImplementedError("not yet implemented")

    def is_distributed(self) -> bool:
        raise NotImplementedError("not yet implemented")
        # return has_xla_support and self._xm.xrt_world_size() > 1

    def finalize(self):
        pass

    @staticmethod
    def create_from_context() -> Optional["_SerialModel"]:
        pass

    @staticmethod
    def create_from_backend(backend: str, **kwargs) -> "_SerialModel":
        pass


registered_computation_models = [_SerialModel, _DistModel, _XlaDistModel]
