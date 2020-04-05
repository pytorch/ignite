import os
import pytest

import torch
import torch.distributed as dist

from ignite.utils import one_rank_only
from ignite.engine import Engine, Events


def _test_distrib_one_rank_only():

    value = torch.tensor([0])

    @one_rank_only(rank=1)
    def initialize():
        value[0] = 100

    initialize()

    value_list = [torch.tensor([0]), torch.tensor([0])]

    dist.all_gather(tensor=value, tensor_list=value_list)

    assert value_list[0].item() == 0
    assert value_list[1].item() == 100


def _test_distrib_one_rank_only_with_engine():

    engine = Engine(lambda e, b: b)

    batch_sum = torch.tensor([0])

    @engine.on(Events.ITERATION_COMPLETED)
    @one_rank_only()  # ie rank == 0
    def _(_):
        batch_sum[0] += engine.state.batch

    engine.run([1, 2, 3], max_epochs=2)

    value_list = [torch.tensor(0), torch.tensor(0)]

    dist.all_gather(tensor=batch_sum[0], tensor_list=value_list)

    assert value_list[0] == 12
    assert value_list[1] == 0


@pytest.mark.distributed
def test_distrib_cpu(distributed_context_single_node_gloo):
    _test_distrib_one_rank_only()
    _test_distrib_one_rank_only_with_engine()


@pytest.mark.multinode_distributed
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    _test_distrib_one_rank_only()
    _test_distrib_one_rank_only_with_engine()
