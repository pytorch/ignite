import tempfile
import shutil

import torch
import torch.distributed as dist

import pytest


@pytest.fixture
def dirname():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


@pytest.fixture()
def local_rank(worker_id):
    """ use a different account in each xdist worker """
    if "gw" in worker_id:
        return int(worker_id.replace("gw", ""))
    elif "master" == worker_id:
        return 0

    raise RuntimeError("Can not get rank from worker_id={}".format(worker_id))


@pytest.fixture()
def distributed_context_single_node_nccl(local_rank):

    import os
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "{}".format(torch.cuda.device_count())

    dist_info = {
        "backend": "nccl",
        "world_size": int(os.environ["WORLD_SIZE"]),
        "rank": local_rank,
        "init_method": "tcp://localhost:2222"
    }

    dist.init_process_group(**dist_info)
    torch.cuda.device(local_rank)

    yield {'local_rank': local_rank}

    # Perform some ops otherwise, next tests fail
    device = "cuda"
    _dummy_dist_op(device)

    dist.destroy_process_group()


@pytest.fixture()
def distributed_context_single_node_gloo(local_rank):

    import os
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"

    dist_info = {
        "backend": "gloo",
        "world_size": int(os.environ["WORLD_SIZE"]),
        "rank": local_rank,
        "init_method": "tcp://localhost:2222"
    }

    dist.init_process_group(**dist_info)

    yield "distributed_context_single_node_gloo"

    # Perform some ops otherwise, next tests fail
    device = "cpu"
    _dummy_dist_op(device)

    dist.destroy_process_group()


@pytest.fixture()
def multi_node_conf(local_rank):
    import os
    assert "node_id" in os.environ
    assert "nnodes" in os.environ
    assert "nproc_per_node" in os.environ

    node_id = int(os.environ['node_id'])
    nnodes = int(os.environ['nnodes'])
    nproc_per_node = int(os.environ['nproc_per_node'])
    out = {
        'world_size': nnodes * nproc_per_node,
        'rank': local_rank + node_id * nproc_per_node,
        'local_rank': local_rank
    }
    return out


@pytest.fixture()
def distributed_context_multi_node_gloo(multi_node_conf):

    import os

    assert "MASTER_ADDR" in os.environ
    assert "MASTER_PORT" in os.environ

    dist_info = {
        "backend": "gloo",
        "init_method": "env://",
        "world_size": multi_node_conf['world_size'],
        "rank": multi_node_conf['rank']
    }

    dist.init_process_group(**dist_info)

    yield multi_node_conf

    # Perform some ops otherwise, next tests fail
    device = "cpu"
    _dummy_dist_op(device)

    dist.destroy_process_group()


@pytest.fixture()
def distributed_context_multi_node_nccl(multi_node_conf):

    import os

    assert "MASTER_ADDR" in os.environ
    assert "MASTER_PORT" in os.environ

    dist_info = {
        "backend": "gloo",
        "init_method": "env://",
        "world_size": multi_node_conf['world_size'],
        "rank": multi_node_conf['rank']
    }

    dist.init_process_group(**dist_info)
    torch.cuda.device(multi_node_conf['local_rank'])

    yield multi_node_conf

    # Perform some ops otherwise, next tests fail
    device = "cuda:{}".format(distributed_context_multi_node_nccl['local_rank'])
    _dummy_dist_op(device)

    dist.destroy_process_group()


def _dummy_dist_op(device):

    def _gather(y):
        output = [torch.zeros_like(y) for i in range(dist.get_world_size())]
        dist.all_gather(output, y)
        y = torch.cat(output, dim=0)
        return y

    y = torch.rand(10, 12, device=device)
    y = _gather(y)
    assert isinstance(y, torch.Tensor)
