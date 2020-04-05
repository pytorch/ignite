import os
import pytest

import torch
import torch.distributed as dist

from ignite.utils import one_rank_only
from ignite.engine import Engine, Events


def _test_distrib_one_rank_only():

    # last rank
    rank = dist.get_world_size() - 1

    value = torch.tensor([0])

    @one_rank_only(rank=rank)
    def initialize():
        value[0] = 100

    initialize()

    value_list = [torch.tensor(0) for _ in range(dist.get_world_size())]

    dist.all_gather(tensor=value, tensor_list=value_list)

    for r in range(dist.get_world_size()):
        if r == rank:
            assert value_list[r].item() == 100
        else:
            assert value_list[r].item() == 0


def _test_distrib_one_rank_only_with_engine():

    engine = Engine(lambda e, b: b)

    batch_sum = torch.tensor([0])

    @engine.on(Events.ITERATION_COMPLETED)
    @one_rank_only()  # ie rank == 0
    def _(_):
        batch_sum[0] += engine.state.batch

    engine.run([1, 2, 3], max_epochs=2)

    value_list = [torch.tensor(0) for _ in range(dist.get_world_size())]

    dist.all_gather(tensor=batch_sum, tensor_list=value_list)

    for r in range(dist.get_world_size()):
        if r == 0:
            assert value_list[r].item() == 12
        else:
            assert value_list[r].item() == 0


@pytest.mark.distributed
def test_distrib_cpu(distributed_context_single_node_gloo):
    _test_distrib_one_rank_only()
    _test_distrib_one_rank_only_with_engine()


@pytest.mark.multinode_distributed
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    _test_distrib_one_rank_only()
    _test_distrib_one_rank_only_with_engine()
