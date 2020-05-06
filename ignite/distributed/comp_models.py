from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import os
import subprocess

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ignite.distributed.utils import has_xla_support


class ComputationModel(metaclass=ABCMeta):
    @abstractmethod
    def is_initialized(self) -> bool:
        pass

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
    def get_nnodes(self) -> int:
        pass

    @abstractmethod
    def get_node(self) -> int:
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
    @staticmethod
    def spawn(*args, **kwargs):
        pass


class _SerialModel(ComputationModel):

    name = "serial"
    available_backends = tuple()

    def is_initialized(self) -> bool:
        return True

    def get_local_rank(self) -> int:
        return 0

    def get_rank(self) -> int:
        return 0

    def get_world_size(self) -> int:
        return 1

    def get_ntasks_per_node(self) -> int:
        return 1

    def get_nnodes(self) -> int:
        return 1

    def get_node(self) -> int:
        return 0

    def is_distributed(self) -> bool:
        return False

    def device(self) -> Union[torch.device, str]:
        return "cpu"

    def backend(self) -> Optional[str]:
        return None

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

    def __init__(self, backend, timeout=None, **kwargs):
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
        self._is_initialized = True

    def _compute_ntasks_per_node(self):
        tensor = torch.tensor([self.get_local_rank() + 1]).to(self.device())
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor.item()

    def setup_env_vars(self):
        if "SLURM_JOB_ID" in os.environ:
            self._setup_env_in_slurm()
            return

        for k in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
            if k not in os.environ:
                raise RuntimeError("PyTorch distributed configuration is missing '{}' in env variables".format(k))

        os.environ["RANK"] = os.environ["RANK"]
        os.environ["LOCAL_RANK"] = os.environ["LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["WORLD_SIZE"]
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

    def is_initialized(self) -> bool:
        return self._is_initialized

    def get_local_rank(self) -> int:
        return self._local_rank

    def get_rank(self) -> int:
        return dist.get_rank()

    def get_world_size(self) -> int:
        return dist.get_world_size()

    def get_ntasks_per_node(self) -> int:
        return self._ntasks_per_node

    def get_nnodes(self) -> int:
        return self._nnodes

    def get_node(self) -> int:
        return self._node

    def is_distributed(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def device(self) -> Union[torch.device, str]:
        if self.backend() == "nccl":
            return "cuda:{}".format(torch.cuda.current_device())
        return "cpu"

    def backend(self) -> Optional[str]:
        return dist.get_backend()

    @staticmethod
    def _dist_worker_task_fn(fn,):

        pass

    @staticmethod
    def spawn(fn, args, num_workers_per_machine, num_machines, machine_rank, dist_url=None, timeout=None, **kwargs):
        world_size = num_machines * num_workers_per_machine
        mp.spawn(
            _DistModel._dist_worker_task_fn,
            nprocs=num_workers_per_machine,
            args=(fn, world_size, num_gpus_per_machine, machine_rank, dist_url, args),
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

    def is_initialized(self) -> bool:
        return True

    def get_local_rank(self) -> int:
        self._xm.get_ordinal()

    def get_rank(self) -> int:
        return self._xm.get_ordinal()

    def get_world_size(self) -> int:
        return self._xm.xrt_world_size()

    def get_ntasks_per_node(self) -> int:
        raise NotImplementedError("not yet implemented")

    def get_nnodes(self) -> int:
        raise NotImplementedError("not yet implemented")

    def get_node(self) -> int:
        raise NotImplementedError("not yet implemented")

    def is_distributed(self) -> bool:
        raise NotImplementedError("not yet implemented")
        # return has_xla_support and self._xm.xrt_world_size() > 1


registered_computation_models = [_SerialModel, _DistModel, _XlaDistModel]
