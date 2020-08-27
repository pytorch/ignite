import os
import subprocess
import warnings
from distutils.version import LooseVersion
from typing import Callable, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ignite.distributed.comp_models.base import ComputationModel

has_native_dist_support = dist.is_available()


if has_native_dist_support:

    NCCL = dist.Backend.NCCL
    GLOO = dist.Backend.GLOO
    MPI = dist.Backend.MPI

    class _NativeDistModel(ComputationModel):
        """Private class for PyTorch native distributed computation model.

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
            name for name in [NCCL, GLOO, MPI] if getattr(dist, "is_{}_available".format(name))()
        )

        @staticmethod
        def create_from_context() -> Optional["_NativeDistModel"]:
            if not (dist.is_available() and dist.is_initialized()):
                return None
            return _NativeDistModel()

        @staticmethod
        def create_from_backend(backend: str, **kwargs) -> "_NativeDistModel":
            if dist.is_available() and dist.is_initialized():
                raise RuntimeError("Can not create new distributed process group if default one is already initialized")
            return _NativeDistModel(backend=backend, **kwargs)

        def __init__(self, backend=None, timeout=None, **kwargs):
            """This is a private method. Please, use `create_from_backend` or `create_from_context`
            """
            super(_NativeDistModel, self).__init__()
            self._env_backup = None
            if backend is not None:
                self._create_from_backend(backend, timeout=timeout, **kwargs)
            else:
                self._init_from_context()

        def _create_from_backend(self, backend, timeout=None, **kwargs):
            if backend == dist.Backend.NCCL and not torch.cuda.is_available():
                raise RuntimeError("Nccl backend is required but no cuda capable devices")

            self.setup_env_vars()

            self._local_rank = int(os.environ["LOCAL_RANK"])
            # for debug purposes
            self._master_port = int(os.environ["MASTER_PORT"])
            self._master_addr = os.environ["MASTER_ADDR"]

            init_pg_kwargs = {}
            if timeout is not None:
                init_pg_kwargs["timeout"] = timeout

            dist.init_process_group(backend, init_method="env://", **init_pg_kwargs)
            # https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
            dist.barrier()

            if backend == dist.Backend.NCCL:
                torch.cuda.set_device(self._local_rank)

            self._setup_attrs()

        def _init_from_context(self):

            self._identify_local_rank()

            # for debug purposes
            self._master_port = None
            self._master_addr = None
            self._setup_attrs()

        def _compute_nproc_per_node(self):
            tensor = torch.tensor([self.get_local_rank() + 1]).to(self.device())
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            return tensor.item()

        def _get_all_hostnames(self):
            import socket

            device = "cpu"
            if self.backend() == dist.Backend.NCCL:
                index = torch.cuda.current_device()
                device = "cuda:{}".format(index)
            name = socket.gethostname()
            name = torch.tensor(bytearray(name, "utf-8")).to(device)
            padded_t_name = torch.zeros(256, device=device, dtype=torch.long)
            padded_t_name[: len(name)] = name
            out_t_names = [torch.zeros_like(padded_t_name) for _ in range(self.get_world_size())]
            dist.all_gather(out_t_names, padded_t_name)
            out_t_names = [tuple(t.cpu().tolist()) for t in out_t_names]
            return out_t_names

        @staticmethod
        def _compute_node_and_local_ranks(rank, hostnames):
            from collections import Counter

            c = Counter(hostnames)
            sizes = torch.tensor([0,] + list(c.values()))
            cumsum_sizes = torch.cumsum(sizes, dim=0)
            node_rank = (rank // cumsum_sizes[1:]).clamp(0, 1).sum().item()
            local_rank = rank - cumsum_sizes[node_rank].item()
            return local_rank, node_rank

        def _compute_local_rank_via_hostname(self):
            # get all hostnames
            hostnames = self._get_all_hostnames()
            local_rank, self._node = self._compute_node_and_local_ranks(self.get_rank(), hostnames)

            if local_rank < 0 or self._node < 0:
                raise ValueError(
                    "Failed to correctly estimate local rank. "
                    "Debugging info: local rank: {}, node rank: {}, hostnames: {}".format(
                        local_rank, self._node, hostnames
                    )
                )
            return local_rank

        def _identify_local_rank(self):

            if "SLURM_JOBID" in os.environ:
                os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]

            if "LOCAL_RANK" in os.environ:
                self._local_rank = int(os.environ["LOCAL_RANK"])
            elif self._ext_local_rank is not None:
                self._local_rank = self._ext_local_rank
            else:
                warnings.warn(
                    "Local rank information for native distributed setting will be initialized using "
                    "heuristic approach based on hostname which can be different of real setup. Please, "
                    "either set `os.environ['LOCAL_RANK']` "
                    "or use `idist.set_local_rank(local_rank)` with correct local rank index."
                )
                # use socket gethostname heuristic to determine number of nodes => local rank
                self._local_rank = self._compute_local_rank_via_hostname()

        def setup_env_vars(self):

            self._env_backup = os.environ.copy()

            if "SLURM_JOBID" in os.environ:
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

        def get_local_rank(self) -> int:
            return self._local_rank

        def get_rank(self) -> int:
            return dist.get_rank()

        def get_world_size(self) -> int:
            return dist.get_world_size()

        def get_nproc_per_node(self) -> int:
            return self._nproc_per_node

        def get_nnodes(self) -> int:
            return self._nnodes

        def get_node_rank(self) -> int:
            return self._node

        def device(self) -> torch.device:
            if self.backend() == dist.Backend.NCCL:
                index = torch.cuda.current_device()
                return torch.device("cuda:{}".format(index))
            return torch.device("cpu")

        def backend(self) -> str:
            return dist.get_backend()

        def finalize(self):
            dist.destroy_process_group()
            # restore backed-up env
            if self._env_backup is not None:
                os.environ.clear()
                os.environ.update(self._env_backup)

        @staticmethod
        def _dist_worker_task_fn(
            local_rank, backend, fn, args, kw_dict, world_size, nprocs_per_node, node_rank, master_addr, master_port, kw
        ):
            from ignite.distributed.utils import _set_model, finalize

            copy_env_vars = os.environ.copy()

            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ["RANK"] = str(node_rank * nprocs_per_node + local_rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["MASTER_ADDR"] = str(master_addr)
            os.environ["MASTER_PORT"] = str(master_port)

            model = _NativeDistModel.create_from_backend(backend, **kw)
            _set_model(model)
            fn(local_rank, *args, **kw_dict)
            finalize()

            os.environ.clear()
            os.environ.update(copy_env_vars)

        @staticmethod
        def spawn(
            fn: Callable,
            args: Tuple,
            kwargs_dict: Optional[Mapping] = None,
            nproc_per_node: int = 1,
            nnodes: int = 1,
            node_rank: int = 0,
            master_addr: str = "127.0.0.1",
            master_port: int = 2222,
            backend: str = "nccl",
            **kwargs
        ):
            world_size = nnodes * nproc_per_node

            spawn_kwargs = {
                "join": kwargs.get("join", True),
                "daemon": kwargs.get("daemon", False),
            }
            # start_method in pytorch >= 1.5
            if LooseVersion(torch.__version__) >= LooseVersion("1.5.0"):
                spawn_kwargs["start_method"] = kwargs.get("start_method", "spawn")

            mp.spawn(
                _NativeDistModel._dist_worker_task_fn,
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

        def _do_all_reduce(self, tensor: torch.Tensor, op: str = "SUM") -> torch.Tensor:
            if op not in self._reduce_op_map:
                raise ValueError("Unsupported reduction operation: '{}'".format(op))
            op = self._reduce_op_map[op]
            dist.all_reduce(tensor, op)
            return tensor

        def _do_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
            if tensor.ndimension() == 0:
                tensor = tensor.unsqueeze(0)
            output = [torch.zeros_like(tensor) for _ in range(self.get_world_size())]
            dist.all_gather(output, tensor)
            return torch.cat(output, dim=0)

        def _do_broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
            dist.broadcast(tensor, src=src)
            return tensor

        def barrier(self):
            dist.barrier()
