import os
import re
import subprocess
import warnings
from distutils.version import LooseVersion
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, cast

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
        - GLOO <-> CPU or GPU
        - MPI  <-> CPU

        """

        name = "native-dist"
        available_backends = tuple(name for name in [NCCL, GLOO, MPI] if getattr(dist, f"is_{name}_available")())

        @staticmethod
        def create_from_context() -> Optional["_NativeDistModel"]:
            if not (dist.is_available() and dist.is_initialized()):
                return None
            return _NativeDistModel()

        @staticmethod
        def create_from_backend(
            backend: str,
            init_method: Optional[str] = None,
            world_size: Optional[int] = None,
            rank: Optional[int] = None,
            **kwargs: Any,
        ) -> "_NativeDistModel":
            if backend not in _NativeDistModel.available_backends:
                raise ValueError(f"Backend should be one of '{_NativeDistModel.available_backends}'")

            if dist.is_available() and dist.is_initialized():
                raise RuntimeError("Can not create new distributed process group if default one is already initialized")

            if init_method is None:
                if world_size is not None or rank is not None:
                    raise ValueError("Arguments rank and world_size should be None if no init_method is provided")
            else:
                has_rank = rank is not None
                has_ws = world_size is not None
                if (has_rank or has_ws) and (not has_rank or not has_ws):
                    raise ValueError(f"Both rank and world_size should be provided, but given {rank} and {world_size}")

            return _NativeDistModel(
                backend=backend, init_method=init_method, world_size=world_size, rank=rank, **kwargs
            )

        def __init__(
            self,
            backend: Optional[str] = None,
            timeout: Optional[int] = None,
            init_method: Optional[str] = None,
            world_size: Optional[int] = None,
            rank: Optional[int] = None,
            **kwargs: Any,
        ) -> None:
            """This is a private method. Please, use `create_from_backend` or `create_from_context`
            """
            super(_NativeDistModel, self).__init__()
            self._env_backup = None  # type: Optional[Dict[str, str]]
            self._local_rank = None  # type: Optional[int]
            self._master_port = None  # type: Optional[int]
            self._master_addr = None  # type: Optional[str]
            self._init_method = None  # type: Optional[str]
            if backend is not None:
                self._create_from_backend(
                    backend, timeout=timeout, init_method=init_method, world_size=world_size, rank=rank, **kwargs
                )
            else:
                self._init_from_context()

        def _create_from_backend(
            self,
            backend: str,
            timeout: Optional[int] = None,
            init_method: Optional[str] = None,
            world_size: Optional[int] = None,
            rank: Optional[int] = None,
            **kwargs: Any,
        ) -> None:
            if backend == dist.Backend.NCCL and not torch.cuda.is_available():
                raise RuntimeError("Nccl backend is required but no cuda capable devices")
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

            if torch.cuda.is_available():
                torch.cuda.set_device(self._local_rank)

            self._setup_attrs()

        def _init_from_context(self) -> None:
            self._backend = dist.get_backend()
            self._identify_local_rank()
            self._setup_attrs()

        def _compute_nproc_per_node(self) -> int:
            local_rank = self.get_local_rank()
            device = torch.device("cpu")
            if torch.cuda.is_available():
                # we manually set cuda device to local rank in order to avoid a hang on all_reduce
                device = torch.device(f"cuda:{local_rank}")
            tensor = torch.tensor([self.get_local_rank() + 1]).to(device)
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            return int(tensor.item())

        def _get_all_hostnames(self) -> List[Tuple[str, ...]]:
            import socket

            device = "cpu"
            if torch.cuda.is_available():
                index = torch.cuda.current_device()
                device = f"cuda:{index}"
            hostname = socket.gethostname()
            name = torch.tensor(bytearray(hostname, "utf-8")).to(device)
            padded_t_name = torch.zeros(256, device=device, dtype=torch.long)
            padded_t_name[: len(name)] = name
            out_t_names = [torch.zeros_like(padded_t_name) for _ in range(self.get_world_size())]
            dist.all_gather(out_t_names, padded_t_name)
            return [tuple(t.cpu().tolist()) for t in out_t_names]

        @staticmethod
        def _compute_node_and_local_ranks(rank: int, hostnames: List[Tuple[str, ...]]) -> Tuple[int, int]:
            from collections import Counter

            c = Counter(hostnames)  # type: Counter
            sizes = torch.tensor([0,] + list(c.values()))
            cumsum_sizes = torch.cumsum(sizes, dim=0)
            node_rank = (rank // cumsum_sizes[1:]).clamp(0, 1).sum().item()
            local_rank = rank - cumsum_sizes[node_rank].item()
            return int(local_rank), node_rank

        def _compute_local_rank_via_hostname(self) -> int:
            # get all hostnames
            hostnames = self._get_all_hostnames()
            local_rank, self._node = self._compute_node_and_local_ranks(self.get_rank(), hostnames)

            if local_rank < 0 or self._node < 0:
                raise ValueError(
                    "Failed to correctly estimate local rank. "
                    f"Debugging info: local rank: {local_rank}, node rank: {self._node}, hostnames: {hostnames}"
                )
            return local_rank

        def _identify_local_rank(self) -> None:

            if "SLURM_JOB_ID" in os.environ:
                os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]

            if "LOCAL_RANK" in os.environ:
                self._local_rank = int(os.environ["LOCAL_RANK"])
            elif self._ext_local_rank is not None:
                self._local_rank = self._ext_local_rank
            else:
                warnings.warn(
                    "Local rank information for native distributed setting will be initialized using "
                    "a heuristic approach based on the hostnames. In some corner cases, determined "
                    "local rank can be different from the real setup. To avoid this warning, "
                    "please either set `os.environ['LOCAL_RANK']` "
                    "or use `idist.set_local_rank(local_rank)` with correct local rank index."
                )
                # use socket gethostname heuristic to determine number of nodes => local rank
                self._local_rank = self._compute_local_rank_via_hostname()

        def setup_env_vars(self, rank: Optional[int] = None, world_size: Optional[int] = None) -> None:

            self._env_backup = os.environ.copy()

            # check whether all necessary env vars are set or not
            env_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
            all_env_vars_defined = [k in os.environ for k in env_vars]

            if "SLURM_JOB_ID" in os.environ:
                if any(all_env_vars_defined):
                    raise RuntimeError(
                        f"Defined env variables '{env_vars}' should not be specified with SLURM. Typically, this "
                        "happens when `torch.distributed.launch` or `torch.multiprocessing.spawn` are used. Please be "
                        "sure to use the `srun` command instead."
                    )
                if rank is not None or world_size is not None:
                    raise ValueError("Arguments rank and world_size should not be specified with SLURM")
                self._setup_env_in_slurm()
            else:
                # check if all necessary env vars are set
                # if partially defined raise an error
                if any(all_env_vars_defined) and not all(all_env_vars_defined):
                    raise RuntimeError(f"PyTorch distributed configuration should define env variables '{env_vars}'")

                os.environ["RANK"] = os.environ.get("RANK", f"{rank if rank is not None else 0}")
                os.environ["WORLD_SIZE"] = os.environ.get(
                    "WORLD_SIZE", f"{world_size if world_size is not None else 1}"
                )
                os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
                os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "15000")
                os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")

            self._local_rank = int(os.environ["LOCAL_RANK"])
            self._master_addr = os.environ["MASTER_ADDR"]
            self._master_port = int(os.environ["MASTER_PORT"])

        def _setup_env_in_slurm(self) -> None:
            for k in ["SLURM_JOB_ID", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS", "SLURM_JOB_NODELIST"]:
                if k not in os.environ:
                    raise RuntimeError(f"SLURM distributed configuration is missing '{k}' in env variables")

            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            # port should be the same over all process
            slurm_port = os.environ["SLURM_JOB_ID"]
            slurm_port = slurm_port[-4:]
            os.environ["MASTER_PORT"] = str(int(slurm_port) + 15000)
            try:
                # use scontrol to expand hostname list
                hostnames = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]])
            except FileNotFoundError:
                # expand hostname list as scontrol
                hostnames = " ".join(_expand_hostlist(os.environ["SLURM_JOB_NODELIST"])).encode("utf-8")
            # master address is the first hostname of nodes list
            os.environ["MASTER_ADDR"] = hostnames.split()[0].decode("utf-8")

        def get_local_rank(self) -> int:
            return cast(int, self._local_rank)

        def get_rank(self) -> int:
            return dist.get_rank()

        def get_world_size(self) -> int:
            return dist.get_world_size()

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
                return torch.device(f"cuda:{index}")
            return torch.device("cpu")

        def backend(self) -> str:
            return dist.get_backend()

        def finalize(self) -> None:
            dist.destroy_process_group()
            # restore backed-up env
            self._restore_env()

        def _restore_env(self) -> None:
            # restore backed-up env
            if self._env_backup is not None:
                os.environ.clear()
                os.environ.update(self._env_backup)

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
                arg_world_size = None
                arg_rank = None

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
            master_addr: Optional[str] = None,
            master_port: Optional[int] = None,
            backend: str = "nccl",
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

        def _do_all_reduce(self, tensor: torch.Tensor, op: str = "SUM") -> torch.Tensor:
            if op not in self._reduce_op_map:
                raise ValueError(f"Unsupported reduction operation: '{op}'")
            reduce_op = self._reduce_op_map[op]
            dist.all_reduce(tensor, reduce_op)
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

        def barrier(self) -> None:
            dist.barrier()

    def _expand_hostlist(nodelist: str) -> List[str]:
        """Expand a compressed hostlist string and returns all hosts listed.

        Source : https://github.com/LLNL/py-hostlist/blob/master/hostlist/hostlist.py

        Args:
            nodelist: Compressed hostlist string

        .. versionadded:: 0.4.6
        """
        node_list = nodelist.split(", ")

        result_hostlist = []
        for node in node_list:
            nodelist_match = r"(\w+-?)\[((,?[0-9]+-?,?-?){0,})\](.*)?"
            if re.search(nodelist_match, node):
                match = re.search(nodelist_match, node)

                if match is None:
                    raise ValueError(f"hostlist unvalid : {nodelist}")

                # holds the ranges of nodes as a string
                # now we can manipulate the string and cast it to a list of numbers
                num = str(match.group(2)).replace("[", "").replace("]", "")

                if len(num) == 0:
                    raise ValueError(f"hostlist unvalid : {nodelist}")

                num_list = num.split(",")

                # find range of node numbers
                ranges = [elem.split("-") for elem in num_list if "-" in elem]

                # if the node numbers contain leading zeros, store them to be
                if len(ranges):
                    lead_zeros = max([len(s) - len(s.lstrip("0")) for s, _ in ranges])
                else:
                    lead_zeros = 0

                # list of expanded ranges of node numbers
                nodes_list = [list(range(int(s), int(e) + 1)) for s, e in ranges]

                # add list of single node numbers
                nodes_list += [[int(elem)] for elem in num_list if "-" not in elem]

                # flat the list
                final_list = [item for sublist in nodes_list for item in sublist]

                # put final list in ascending order and append cluster name to each node number
                final_list = list(set(sorted(final_list)))

                # prepend leading zeros to numbers required
                hostlist_tmp = [str(elem).zfill(lead_zeros + 1) for elem in final_list]

                # append hostname to the node numbers
                hostlist_no_suffix = [match.group(1) + elem for elem in hostlist_tmp]

                # append suffix to hostlist if there is one
                final_hostlist = [elem + match.group(4) for elem in hostlist_no_suffix]

                result_hostlist += final_hostlist
            else:
                result_hostlist.append(node)

        return result_hostlist
