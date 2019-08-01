import pytest

import torch
import torch.distributed as dist


@pytest.fixture()
def local_rank(worker_id):
    """ use a different account in each xdist worker """
    return int(worker_id.replace("gw", ""))


@pytest.fixture()
def distributed_context_single_node(local_rank):
    # import os
    # os.environ["WORLD_SIZE"] = "{}".format(torch.cuda.device_count())
    # os.environ["RANK"] = "{}".format(local_rank)

    dist_info = {
        "backend": "nccl",
        "world_size": torch.cuda.device_count(),
        "rank": local_rank,
        "init_method": "tcp://localhost:2222"
    }

    g = dist.init_process_group(**dist_info)

    yield g

    dist.destroy_process_group()
