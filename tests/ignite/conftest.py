import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist


@pytest.fixture()
def dirname():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


@pytest.fixture()
def get_rank_zero_dirname(dirname):
    def func(device):
        import ignite.distributed as idist

        zero_rank_dirname = idist.all_gather(dirname)[0]
        return zero_rank_dirname

    yield func


@pytest.fixture()
def local_rank(worker_id):
    """ use a different account in each xdist worker """
    import os

    if "gw" in worker_id:
        lrank = int(worker_id.replace("gw", ""))
    elif "master" == worker_id:
        lrank = 0
    else:
        raise RuntimeError("Can not get rank from worker_id={}".format(worker_id))

    os.environ["LOCAL_RANK"] = "{}".format(lrank)

    yield lrank

    del os.environ["LOCAL_RANK"]


@pytest.fixture()
def world_size():
    import os

    remove_env_var = False

    if "WORLD_SIZE" not in os.environ:
        if torch.cuda.is_available():
            ws = torch.cuda.device_count()
        else:
            ws = 1
        os.environ["WORLD_SIZE"] = "{}".format(ws)
        remove_env_var = True

    yield int(os.environ["WORLD_SIZE"])

    if remove_env_var:
        del os.environ["WORLD_SIZE"]


@pytest.fixture()
def clean_env():
    import os

    for k in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
        if k in os.environ:
            del os.environ[k]


def _create_dist_context(dist_info, lrank):

    dist.init_process_group(**dist_info)
    dist.barrier()
    if dist_info["backend"] == "nccl":
        torch.cuda.set_device(lrank)

    return {"local_rank": lrank}


def _destroy_dist_context():

    dist.barrier()
    dist.destroy_process_group()

    from ignite.distributed.utils import _SerialModel, _set_model

    # We need to set synced model to initial state
    _set_model(_SerialModel())


@pytest.fixture()
def distributed_context_single_node_nccl(local_rank, world_size):

    dist_info = {
        "backend": "nccl",
        "world_size": world_size,
        "rank": local_rank,
        "init_method": "tcp://localhost:2223",
    }
    yield _create_dist_context(dist_info, local_rank)
    _destroy_dist_context()


@pytest.fixture()
def distributed_context_single_node_gloo(local_rank, world_size):

    from datetime import timedelta

    dist_info = {
        "backend": "gloo",
        "world_size": world_size,
        "rank": local_rank,
        "init_method": "tcp://localhost:2222",
        "timeout": timedelta(seconds=60),
    }
    yield _create_dist_context(dist_info, local_rank)
    _destroy_dist_context()


@pytest.fixture()
def multi_node_conf(local_rank):
    import os

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
    if dist_info["backend"] == "nccl":
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

    import os

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

    import os

    assert "MASTER_ADDR" in os.environ
    assert "MASTER_PORT" in os.environ

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
    import os
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
