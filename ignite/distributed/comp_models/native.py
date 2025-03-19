import os
import re
import subprocess
import warnings
from typing import Any, Callable, cast, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from packaging.version import Version

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
            """This is a private method. Please, use `create_from_backend` or `create_from_context`"""
            super(_NativeDistModel, self).__init__()
            self._env_backup: Optional[Dict[str, str]] = None
            self._local_rank: Optional[int] = None
            self._master_port: Optional[int] = None
            self._master_addr: Optional[str] = None
            self._init_method: Optional[str] = None
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

            init_pg_kwargs: Dict[str, Any] = {}
            if timeout is not None:
                init_pg_kwargs["timeout"] = timeout

            if init_method is None:
                init_method = "env://"

            if "env" not in init_method:
                init_pg_kwargs["world_size"] = int(os.environ["WORLD_SIZE"])
                init_pg_kwargs["rank"] = int(os.environ["RANK"])
            self._init_method = init_method

            dist.init_process_group(backend, init_method=init_method, **init_pg_kwargs)

            if torch.cuda.is_available():
                torch.cuda.set_device(self._local_rank)

            # Call barrier after init_process_group as in
            # https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
            # Define device ids for NCCL to avoid warnings
            # [W ProcessGroupNCCL.cpp:1569] Rank 0 using best-guess GPU 0 to perform barrier as devices used by
            # this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping
            # is incorrect.Specify device_ids in barrier() to force use of a particular device.
            if backend == dist.Backend.NCCL and Version(torch.__version__) >= Version("1.8.0"):
                device_ids = [torch.cuda.current_device()]
                dist.barrier(device_ids=device_ids)
            else:
                # For older versions there is no device_ids arg
                dist.barrier()

            self._setup_attrs()

        def _init_from_context(self) -> None:
            self._backend = dist.get_backend()
            self._identify_local_rank()
            self._setup_attrs()

        def _compute_nproc_per_node(self) -> int:
            local_rank = self.get_local_rank()
            # Create new cpu group to get nproc_per_node such we avoid using
            # badly configured NCCL
            gloo_group = dist.new_group(backend="gloo")
            tensor = torch.tensor([local_rank + 1]).to("cpu")
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=gloo_group)
            dist.destroy_process_group(gloo_group)
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

            c: Counter = Counter(hostnames)
            sizes = torch.tensor([0] + list(c.values()))
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

            if "SLURM_JOB_ID" in os.environ:
                if rank is not None or world_size is not None:
                    raise ValueError("Arguments rank and world_size should not be specified with SLURM")
                self._setup_env_in_slurm()
            else:
                env_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
                all_env_vars_defined = [k in os.environ for k in env_vars]
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
            slurm_env_req_vars = [
                "SLURM_JOB_ID",
                "SLURM_PROCID",
                "SLURM_LOCALID",
                "SLURM_NTASKS",
                "SLURM_JOB_NODELIST",
                "SLURM_JOB_NUM_NODES",
            ]
            for k in slurm_env_req_vars:
                if k not in os.environ:
                    raise RuntimeError(f"SLURM distributed configuration is missing '{k}' in env variables")

            ddp_vars = _setup_ddp_vars_from_slurm_env(cast(Dict, os.environ))

            # define DDP env vars required by PTH:
            for key, value in ddp_vars.items():
                os.environ[key] = str(value)

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

            arg_world_size: Optional[int] = world_size
            arg_rank: Optional[int] = rank
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
        def spawn(
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
            if Version(torch.__version__) >= Version("1.5.0"):
                import builtins

                if "__IPYTHON__" in builtins.__dict__:
                    # use fork in jupyter
                    default_start_method = "fork"
                else:
                    default_start_method = "spawn"
                spawn_kwargs["start_method"] = kwargs.get("start_method", default_start_method)
                start_processes = mp.start_processes
            # TODO: `spawn` wrongfully does not adopt address and port from environment if `init_method` is "env://"
            if init_method in [None, "env://"]:
                init_method = "env://"
                if master_port is None:
                    master_port = 2222
                if master_addr is None:
                    master_addr = "127.0.0.1"
            elif master_addr is not None:
                raise ValueError("master_addr should be None if init_method is provided other then 'env://'")
            elif master_port is not None:
                raise ValueError("master_port should be None if init_method is provided other then 'env://'")

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

        def _setup_group(self, group: Any) -> dist.ProcessGroup:
            if isinstance(group, list) and all(isinstance(item, int) for item in group):
                group = self._do_new_group(group)
            if not (isinstance(group, dist.ProcessGroup) or group == dist.GroupMember.NON_GROUP_MEMBER):
                raise ValueError(
                    f"Argument group should be list of int or ProcessGroup, got {type(group)}, group={group}"
                )
            return group

        _reduce_op_map = {
            "SUM": dist.ReduceOp.SUM,
            "PRODUCT": dist.ReduceOp.PRODUCT,
            "MIN": dist.ReduceOp.MIN,
            "MAX": dist.ReduceOp.MAX,
            "AND": dist.ReduceOp.BAND,
            "OR": dist.ReduceOp.BOR,
        }

        def _do_all_reduce(self, tensor: torch.Tensor, op: str = "SUM", group: Optional[Any] = None) -> torch.Tensor:
            if op not in self._reduce_op_map:
                raise ValueError(f"Unsupported reduction operation: '{op}'")
            if group is not None:
                group = self._setup_group(group)
            reduce_op = self._reduce_op_map[op]
            # We do if/else here for compatibility with older pytorch versions
            if group is not None:
                dist.all_reduce(tensor, reduce_op, group=group)
            else:
                dist.all_reduce(tensor, reduce_op)
            return tensor

        def _do_all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None) -> torch.Tensor:
            if group is not None:
                group = self._setup_group(group)
            if self._rank_not_in_group(group):
                return tensor
            if group is None:
                group_size = self.get_world_size()
            else:
                group_size = group.size()
            if tensor.ndimension() == 0:
                tensor = tensor.unsqueeze(0)
            output = [torch.zeros_like(tensor) for _ in range(group_size)]
            # We do if/else here for compatibility with older pytorch versions
            if group is not None:
                dist.all_gather(output, tensor, group=group)
            else:
                dist.all_gather(output, tensor)
            return torch.cat(output, dim=0)

        def _do_all_gather_object(self, tensor: Any, group: Optional[Any] = None) -> List[Any]:
            if Version(torch.__version__) < Version("1.7.0"):
                raise RuntimeError(
                    "Current torch version does not implement dist.all_gather_object. "
                    "Required version should be >=1.7.0"
                )
            if group is not None:
                group = self._setup_group(group)
            if self._rank_not_in_group(group):
                return tensor
            if group is None:
                group_size = self.get_world_size()
            else:
                group_size = group.size()
            output = [None for _ in range(group_size)]
            # We do if/else here for compatibility with older pytorch versions
            if group is not None:
                dist.all_gather_object(output, tensor, group=group)
            else:
                dist.all_gather_object(output, tensor)

            return output

        def _do_new_group(self, ranks: List[int], **kwargs: Any) -> Any:
            return dist.new_group(ranks=ranks, **kwargs)

        def _do_broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
            dist.broadcast(tensor, src=src)
            return tensor

        def barrier(self) -> None:
            dist.barrier()

        def _rank_not_in_group(self, group: Optional[Any]) -> bool:
            return dist._rank_not_in_group(group)

    def _expand_hostlist(nodelist: str) -> List[str]:
        """Expand a compressed hostlist string and returns all hosts listed.

        Source : https://github.com/LLNL/py-hostlist/blob/master/hostlist/hostlist.py

        Args:
            nodelist: Compressed hostlist string

        .. note::
            The host names can be composed by any character except the special ones `[`, `]`, `,`. Only one
            sequence `[...]` is supported per hostname.

        .. versionadded:: 0.4.6
        """
        result_hostlist = []

        nodelist_match = r"([^,\[\]]+\[[^\[\]]*\][^,\[\]]*|[^,\[\]]*),?"

        nodelist = nodelist.replace(" ", "")

        for node in re.findall(nodelist_match, nodelist):
            node_match = r"(.+)\[((,?[0-9]+-?,?-?){0,})\](.*)?"

            match = re.search(node_match, node)

            if match is None:
                if node:
                    result_hostlist.append(node)
            else:
                # holds the ranges of nodes as a string
                # now we can manipulate the string and cast it to a list of numbers
                num = str(match.group(2)).replace("[", "").replace("]", "")

                if len(num) == 0:
                    raise ValueError(f"hostlist invalid : {nodelist}")

                num_list = num.split(",")

                # find range of node numbers
                ranges = [elem.split("-") if "-" in elem else [elem, elem] for elem in num_list]

                # if the node numbers contain leading zeros, store them to be
                lead_zeros = max([len(s) - len(s.lstrip("0")) for s, _ in ranges])

                # list of expanded ranges of node numbers
                nodes_list = [list(range(int(s), int(e) + 1)) for s, e in ranges]

                # flat the list
                final_list = [item for sublist in nodes_list for item in sublist]

                # put final list in ascending order and append cluster name to each node number
                final_list = list(sorted(set(final_list)))

                # prepend leading zeros to numbers required
                hostlist_tmp = [str(elem).zfill(lead_zeros + 1) for elem in final_list]

                # append hostname to the node numbers
                hostlist_no_suffix = [match.group(1) + elem for elem in hostlist_tmp]

                # append suffix to hostlist if there is one
                final_hostlist = [elem + match.group(4) for elem in hostlist_no_suffix]

                result_hostlist += final_hostlist

        return result_hostlist

    def _setup_ddp_vars_from_slurm_env(environ: Dict[str, str]) -> Dict[str, Union[str, int]]:
        """Method to setup DDP env vars required by PyTorch from SLURM env"""
        # 1) Tools like enroot can have hooks to translate slurm env vars to RANK, LOCAL_RANK, WORLD_SIZE etc
        # See https://github.com/NVIDIA/enroot/blob/v3.1.0/conf/hooks/extra/50-slurm-pytorch.sh
        # 2) User can use torch.distributed.launch tool to schedule on N local GPUs using 1 node, 1 task by SLURM
        # To cover case 1), let's ensure that defined RANK == SLURM_PROCID, LOCAL_RANK == SLURM_LOCALID,
        #   WORLD_SIZE == SLURM_NTASKS. We will use defined MASTER_ADDR and MASTER_PORT instead of defining
        #   them by our means
        # To cover case 2), let's check that defined RANK >= SLURM_PROCID, LOCAL_RANK >= SLURM_LOCALID,
        #   WORLD_SIZE >= SLURM_NTASKS, SLURM_JOB_NUM_NODES == 1

        ddp_vars: Dict[str, Union[str, int, None]] = {
            "RANK": int(environ["SLURM_PROCID"]),
            "LOCAL_RANK": int(environ["SLURM_LOCALID"]),
            "WORLD_SIZE": int(environ["SLURM_NTASKS"]),
            "MASTER_ADDR": None,
            "MASTER_PORT": None,
        }

        pth_ddp_env_vars = {key: environ.get(key, None) for key in ddp_vars}
        defined_pth_ddp_env_vars = [v is not None for v in pth_ddp_env_vars.values()]
        if all(defined_pth_ddp_env_vars):
            nnodes = int(environ["SLURM_JOB_NUM_NODES"])
            if nnodes > 1:
                # ensure that all pth_ddp_env_vars are consistent with slurm vars
                for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
                    slurm_var = cast(int, ddp_vars[key])
                    pth_var = int(cast(str, pth_ddp_env_vars[key]))
                    if slurm_var != pth_var:
                        raise RuntimeError(
                            "Environment variable defined for PyTorch Distributed context is inconsistent with "
                            f"equivalent SLURM env variable. {key}: {pth_var} vs {slurm_var}\n"
                            f"SLURM vars: {ddp_vars}\n"
                            f"PTH vars: {pth_ddp_env_vars}\n"
                        )
            else:
                # ensure that PTH RANK >= SLURM_PROCID, PTH LOCAL_RANK >= SLURM_LOCALID,
                # PTH WORLD_SIZE >= SLURM_NTASKS
                for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
                    slurm_var = cast(int, ddp_vars[key])
                    pth_var = int(cast(str, pth_ddp_env_vars[key]))
                    if pth_var < slurm_var:
                        raise RuntimeError(
                            "Environment variable defined for PyTorch Distributed context is "
                            "inconsistent with equivalent SLURM env variable. "
                            f"We expect that {key}: {pth_var} >= {slurm_var}\n"
                            f"SLURM vars: {ddp_vars}\n"
                            f"PTH vars: {pth_ddp_env_vars}\n"
                        )
                    ddp_vars[key] = pth_var
            # set up MASTER_ADDR and MASTER_PORT from PTH
            ddp_vars["MASTER_ADDR"] = cast(str, pth_ddp_env_vars["MASTER_ADDR"])
            ddp_vars["MASTER_PORT"] = int(cast(str, pth_ddp_env_vars["MASTER_PORT"]))
        elif any(defined_pth_ddp_env_vars):
            # Let's warn user about PTH env variables that we could not taken into account
            warnings.warn(
                "We detected the following env variables: "
                f"{[(k, v) for k, v in pth_ddp_env_vars.items() if v is not None]},\n"
                "but will not take them into account as the following env vars are missing:"
                f"{[k for k, v in pth_ddp_env_vars.items() if v is None]},\n"
            )

        if ddp_vars["MASTER_ADDR"] is None:
            nodelist = environ["SLURM_JOB_NODELIST"]
            try:
                # use scontrol to expand hostname list
                hostnames = subprocess.check_output(["scontrol", "show", "hostnames", nodelist])
                method = "scontrol"
            except FileNotFoundError:
                # expand hostname list as scontrol
                hostnames = " ".join(_expand_hostlist(nodelist)).encode("utf-8")
                method = "ignite"
            # at least one hostname should be defined
            hostname_list = hostnames.split()
            if len(hostname_list) < 1:
                raise RuntimeError(f"No hostname detected in SLURM_JOB_NODELIST by {method} (nodelist={nodelist})")
            # master address is the first hostname of nodes list
            ddp_vars["MASTER_ADDR"] = str(hostname_list[0].decode("utf-8"))

        if ddp_vars["MASTER_PORT"] is None:
            # port should be the same over all process
            slurm_port = environ["SLURM_JOB_ID"]
            slurm_port = slurm_port[-4:]
            ddp_vars["MASTER_PORT"] = int(slurm_port) + 15000

        return cast(Dict[str, Union[str, int]], ddp_vars)
