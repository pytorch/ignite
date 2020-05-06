import os
import socket
import subprocess

import torch
import torch.distributed as dist

try:
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.parallel_loader import ParallelLoader

    has_xla_support = True
except ImportError:
    has_xla_support = False

from ignite.utils import setup_logger


class SerialModel:
    @staticmethod
    def is_initialized() -> bool:
        return True

    @staticmethod
    def get_local_rank() -> int:
        return 0

    @staticmethod
    def get_rank() -> int:
        return 0

    @staticmethod
    def get_world_size() -> int:
        return 0

    @staticmethod
    def get_ntasks_per_node() -> int:
        return 1

    @staticmethod
    def get_nnodes() -> int:
        return 1

    @staticmethod
    def get_node() -> int:
        return 0

    @staticmethod
    def is_distributed() -> bool:
        return False

    @staticmethod
    def name() -> str:
        return "sequential"


class XlaModel:
    @staticmethod
    def is_initialized() -> bool:
        return True

    @staticmethod
    def get_local_rank() -> int:
        raise Exception("Not yet implemented")

    @staticmethod
    def get_rank() -> int:
        return xm.get_ordinal()

    @staticmethod
    def get_world_size() -> int:
        return xm.xrt_world_size()

    @staticmethod
    def get_ntasks_per_node() -> int:
        raise Exception("not yet implemented")

    @staticmethod
    def get_nnodes() -> int:
        raise Exception("not yet implemented")

    @staticmethod
    def get_node() -> int:
        raise Exception("not yet implemented")

    @staticmethod
    def is_distributed() -> bool:
        return has_xla_support and xm.xrt_world_size() > 1

    @staticmethod
    def name() -> str:
        return "xla"


class DistModel:
    def __init__(self):
        if "SLURM_JOB_ID" in os.environ:
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
        else:
            os.environ["RANK"] = os.environ.get("RANK", "0")
            os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
            os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
            os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
            os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "15000")
        self._master_port = int(os.environ["MASTER_PORT"])
        # for debug
        self._master_addr = os.environ["MASTER_ADDR"]
        self._local_rank = int(os.environ["LOCAL_RANK"])
        # will be defined first time get_nproc_per_node() is called
        self._ntasks_per_node = None
        self._nnodes = None
        self._node = None
        self._is_initialized = False

    def finalize(self):
        # dist must be initialized
        assert dist.is_initialized()
        if self._ntasks_per_node is None:
            tensor = torch.tensor([self.get_local_rank() + 1])
            dist.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
            self._ntasks_per_node = tensor.item()
        if self._nnodes is None:
            self._nnodes = self.get_world_size() // self.get_ntasks_per_node()
        if self._node is None:
            self._node = self.get_rank() // self._ntasks_per_node
        self._is_initialized = True

    def is_initialized(self) -> bool:
        return self._is_initialized

    def get_local_rank(self) -> int:
        assert dist.is_available() and dist.is_initialized()
        return self._local_rank

    @staticmethod
    def get_rank() -> int:
        assert dist.is_available() and dist.is_initialized()
        return dist.get_rank()

    @staticmethod
    def get_world_size() -> int:
        assert dist.is_available() and dist.is_initialized()
        return dist.get_world_size()

    def get_ntasks_per_node(self) -> int:
        return self._ntasks_per_node

    def get_nnodes(self) -> int:
        return self._nnodes

    def get_node(self) -> int:
        return self._node

    @staticmethod
    def is_distributed() -> bool:
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 0

    @staticmethod
    def name() -> str:
        return "dist"


# default is sequential model
_model = SerialModel()
# default backend is None
_backend = None
# the model and backend could change at initialization


def is_distributed() -> bool:
    return _model.is_distributed()


def is_initialized() -> bool:
    return _model.is_initialized()


def model_name() -> str:
    return _model.name()


def world_size() -> int:
    return _model.get_world_size()


def rank() -> int:
    return _model.get_rank()


def local_rank() -> int:
    return _model.get_local_rank()


# experimental
# to be tested with slurm
# it depends on how scheduling was done


def ntasks_per_node() -> int:
    return _model.get_ntasks_per_node()


def nnodes() -> int:
    return _model.get_nnodes()


def node() -> int:
    return _model.get_rank() % _model.get_ntasks_per_node()


def device():
    # nccl means cuda
    if _backend == "nccl":
        return "cuda"
    # gloo means cpu
    elif _backend == "gloo":
        return "cpu"
    # None means
    # SerialModel : cuda is chosen if available otherwise cpu
    # or XlaModel : tpu
    else:
        if _model.name() == "xla":
            return "tpu"
        elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
        else:
            return "cpu"


def backend():
    return _backend


def hostname() -> str:
    return socket.gethostname()


def _find_best_dist_backend():
    assert dist.is_available()
    # select best backend nccl first, gloo second
    if dist.is_nccl_available():
        return "nccl"
    elif dist.is_gloo_available():
        return "gloo"
    else:
        raise Exception("No backend found (missing gloo and nccl")


def sanity_check():
    assert _model.is_initialized()
    assert _model.get_world_size() == _model.get_nnodes() * _model.get_ntasks_per_node()
    assert _model.get_local_rank() < _model.get_ntasks_per_node()
    assert _model.get_rank() < _model.get_world_size()
    assert _model.get_node() < _model.get_nnodes()


def initialize(backend=None):
    global _model, _backend, _device
    if has_xla_support:
        # no backend for tpu
        assert backend is None
        # if device imposed, it should be tpu
        if device is not None:
            assert device == "tpu"
        _device = "tpu"
        _model = XlaModel()
        # how to check if already initialized ?
        xm.rendezvous("init")
    elif dist.is_available():
        if backend is None:
            _backend = _find_best_dist_backend()
        else:
            # imposed by user
            _backend = backend
        assert not dist.is_initialized()
        if backend == "nccl":
            torch.backends.cudnn.benchmark = True
        _model = DistModel()
        # should this be in DistModel ?
        dist.init_process_group(backend, init_method="env://")
        # compute missing information (nnodes, tasks_per_node, etc.)
        _model.finalize()
        if device() == "cuda":
            torch.cuda.set_device(local_rank())
            # number of proc per node should be related to number of GPUs
            assert ntasks_per_node() <= torch.cuda.device_count()
        # TODO: check if backend is available
        # TODO: check device wrt backend (nccl and cpu is not possible)
    # test the configuration
    sanity_check()


def finalize():
    if is_distributed() and model_name() == "dist":
        dist.destroy_process_group()


def show_config():

    # setup parallel logger
    logger = setup_logger(__name__)

    logger.info("model = {}".format(model_name()))
    logger.info("is distributed = {}".format(is_distributed()))
    logger.info("backend = {}".format(backend()))
    logger.info("device = {}".format(device()))
    logger.info("hostname = {}".format(hostname()))
    logger.info("world size = {}".format(world_size()))
    logger.info("rank = {}".format(rank()))
    logger.info("ntasks_per_node = {}".format(ntasks_per_node()))
    logger.info("nnodes = {}".format(nnodes()))
    logger.info("node = {}".format(node()))
