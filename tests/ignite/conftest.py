import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


@pytest.fixture()
def dirname():
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path)


@pytest.fixture()
def get_fixed_dirname(worker_id):
    # multi-proc friendly fixed tmp dirname
    path = "/tmp/fixed_tmp_dirname_"
    lrank = int(worker_id.replace("gw", "")) if "gw" in worker_id else 0

    def getter(name="test"):
        nonlocal path
        path += name
        time.sleep(0.5 * lrank)
        os.makedirs(path, exist_ok=True)
        return path

    yield getter

    time.sleep(1.0 * lrank + 1.0)
    if Path(path).exists():
        shutil.rmtree(path)
    # sort of sync
    time.sleep(1.0)


@pytest.fixture()
def get_rank_zero_dirname(dirname):
    def func():
        import ignite.distributed as idist

        zero_rank_dirname = Path(idist.all_gather(str(dirname))[0])
        return zero_rank_dirname

    yield func


@pytest.fixture()
def local_rank(worker_id):
    """use a different account in each xdist worker"""

    if "gw" in worker_id:
        lrank = int(worker_id.replace("gw", ""))
    elif "master" == worker_id:
        lrank = 0
    else:
        raise RuntimeError(f"Can not get rank from worker_id={worker_id}")

    os.environ["LOCAL_RANK"] = f"{lrank}"

    yield lrank

    del os.environ["LOCAL_RANK"]


@pytest.fixture()
def world_size():

    remove_env_var = False

    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
        remove_env_var = True

    yield int(os.environ["WORLD_SIZE"])

    if remove_env_var:
        del os.environ["WORLD_SIZE"]


@pytest.fixture()
def clean_env():

    for k in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
        if k in os.environ:
            del os.environ[k]


def _create_dist_context(dist_info, lrank):

    dist.init_process_group(**dist_info)
    dist.barrier()
    if torch.cuda.is_available():
        torch.cuda.set_device(lrank)

    return {"local_rank": lrank, "world_size": dist_info["world_size"], "rank": dist_info["rank"]}


def _destroy_dist_context():

    dist.barrier()
    dist.destroy_process_group()

    from ignite.distributed.utils import _SerialModel, _set_model

    # We need to set synced model to initial state
    _set_model(_SerialModel())


def _find_free_port():
    # Taken from https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/launch.py
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _setup_free_port(local_rank):

    port_file = "/tmp/free_port"

    if local_rank == 0:
        port = _find_free_port()
        with open(port_file, "w") as h:
            h.write(str(port))
        return port
    else:
        counter = 10
        while counter > 0:
            counter -= 1
            time.sleep(1)
            if not Path(port_file).exists():
                continue
            with open(port_file, "r") as h:
                port = h.readline()
                return int(port)

        raise RuntimeError(f"Failed to fetch free port on local rank {local_rank}")


@pytest.fixture()
def distributed_context_single_node_nccl(local_rank, world_size):

    free_port = _setup_free_port(local_rank)

    dist_info = {
        "backend": "nccl",
        "world_size": world_size,
        "rank": local_rank,
        "init_method": f"tcp://localhost:{free_port}",
    }
    yield _create_dist_context(dist_info, local_rank)
    _destroy_dist_context()


@pytest.fixture()
def distributed_context_single_node_gloo(local_rank, world_size):

    from datetime import timedelta

    if sys.platform.startswith("win"):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        # can't use backslashes in f-strings
        backslash = "\\"
        init_method = f'file:///{temp_file.name.replace(backslash, "/")}'
    else:
        free_port = _setup_free_port(local_rank)
        init_method = f"tcp://localhost:{free_port}"
        temp_file = None

    dist_info = {
        "backend": "gloo",
        "world_size": world_size,
        "rank": local_rank,
        "init_method": init_method,
        "timeout": timedelta(seconds=60),
    }
    yield _create_dist_context(dist_info, local_rank)
    _destroy_dist_context()
    if temp_file:
        temp_file.close()


@pytest.fixture()
def multi_node_conf(local_rank):

    assert "node_id" in os.environ
    assert "nnodes" in os.environ
    assert "nproc_per_node" in os.environ

    node_id = int(os.environ["node_id"])
    nnodes = int(os.environ["nnodes"])
    nproc_per_node = int(os.environ["nproc_per_node"])
    out = {
        "world_size": nnodes * nproc_per_node,
        "rank": local_rank + node_id * nproc_per_node,
        "local_rank": local_rank,
    }
    return out


def _create_mnodes_dist_context(dist_info, mnodes_conf):

    dist.init_process_group(**dist_info)
    dist.barrier()
    if torch.cuda.is_available():
        torch.cuda.device(mnodes_conf["local_rank"])
    return mnodes_conf


def _destroy_mnodes_dist_context():
    dist.barrier()
    dist.destroy_process_group()

    from ignite.distributed.utils import _SerialModel, _set_model

    # We need to set synced model to initial state
    _set_model(_SerialModel())


@pytest.fixture()
def distributed_context_multi_node_gloo(multi_node_conf):

    assert "MASTER_ADDR" in os.environ
    assert "MASTER_PORT" in os.environ

    dist_info = {
        "backend": "gloo",
        "init_method": "env://",
        "world_size": multi_node_conf["world_size"],
        "rank": multi_node_conf["rank"],
    }
    yield _create_mnodes_dist_context(dist_info, multi_node_conf)
    _destroy_mnodes_dist_context()


@pytest.fixture()
def distributed_context_multi_node_nccl(multi_node_conf):

    assert "MASTER_ADDR" in os.environ
    assert "MASTER_PORT" in os.environ

    os.environ["MASTER_PORT"] = str(int(os.getenv("MASTER_PORT")) + 1)

    dist_info = {
        "backend": "nccl",
        "init_method": "env://",
        "world_size": multi_node_conf["world_size"],
        "rank": multi_node_conf["rank"],
    }
    yield _create_mnodes_dist_context(dist_info, multi_node_conf)
    _destroy_mnodes_dist_context()


def _xla_template_worker_task(index, fn, args):
    import torch_xla.core.xla_model as xm

    xm.rendezvous("init")
    fn(index, *args)


def _xla_execute(fn, args, nprocs):

    import torch_xla.distributed.xla_multiprocessing as xmp

    spawn_kwargs = {}
    if "COLAB_TPU_ADDR" in os.environ:
        spawn_kwargs["start_method"] = "fork"

    try:
        xmp.spawn(_xla_template_worker_task, args=(fn, args), nprocs=nprocs, **spawn_kwargs)
    except SystemExit as ex_:
        assert ex_.code == 0, "Didn't successfully exit in XLA test"


@pytest.fixture()
def xmp_executor():
    yield _xla_execute


@pytest.fixture()
def mock_gpu_is_not_available():
    from unittest.mock import patch

    with patch("torch.cuda") as mock_cuda:
        mock_cuda.is_available.return_value = False
        yield mock_cuda


def _hvd_task_with_init(func, args):
    import horovod.torch as hvd

    hvd.init()
    lrank = hvd.local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(lrank)

    func(*args)

    # Added a sleep to avoid flaky failures on circle ci
    # Sometimes a rank is terminated before final collective
    # op is finished.
    # https://github.com/pytorch/ignite/pull/2357
    time.sleep(2)

    hvd.shutdown()


def _gloo_hvd_execute(func, args, np=1, do_init=False):
    try:
        # old API
        from horovod.run.runner import run
    except ImportError:
        # new API: https://github.com/horovod/horovod/pull/2099
        from horovod import run

    kwargs = dict(use_gloo=True, num_proc=np)

    if do_init:
        return run(_hvd_task_with_init, args=(func, args), **kwargs)

    return run(func, args=args, **kwargs)


@pytest.fixture()
def gloo_hvd_executor():
    yield _gloo_hvd_execute
