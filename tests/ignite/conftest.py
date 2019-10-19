import pytest

import torch
import torch.distributed as dist


@pytest.fixture()
def local_rank(worker_id):
    """ use a different account in each xdist worker """
    if "gw" in worker_id:
        return int(worker_id.replace("gw", ""))
    elif "master" == worker_id:
        return 0
    return worker_id


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

    g = dist.init_process_group(**dist_info)
    torch.cuda.device(local_rank)

    yield g

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

    g = dist.init_process_group(**dist_info)

    yield g

    dist.destroy_process_group()
