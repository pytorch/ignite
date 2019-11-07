import os

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import RootMeanSquaredError

import pytest


def test_zero_div():
    rmse = RootMeanSquaredError()
    with pytest.raises(NotComputableError):
        rmse.compute()


def test_compute():
    rmse = RootMeanSquaredError()

    y_pred = torch.Tensor([[2.0], [-2.0]])
    y = torch.zeros(2)
    rmse.update((y_pred, y))
    assert isinstance(rmse.compute(), float)
    assert rmse.compute() == 2.0

    rmse.reset()
    y_pred = torch.Tensor([[3.0], [-3.0]])
    y = torch.zeros(2)
    rmse.update((y_pred, y))
    assert isinstance(rmse.compute(), float)
    assert rmse.compute() == 3.0


def _test_distrib_itegration(device):
    import numpy as np
    import torch.distributed as dist
    from ignite.engine import Engine

    rank = dist.get_rank()
    n_iters = 100
    s = 50
    offset = n_iters * s

    y_true = torch.arange(0, offset * dist.get_world_size(), dtype=torch.float).to(device)
    y_preds = (rank + 1) * torch.ones(offset, dtype=torch.float).to(device)

    def update(engine, i):
        return y_preds[i * s:(i + 1) * s], y_true[i * s + offset * rank:(i + 1) * s + offset * rank]

    engine = Engine(update)

    m = RootMeanSquaredError(device=device)
    m.attach(engine, "rmse")

    data = list(range(n_iters))
    engine.run(data=data, max_epochs=1)

    assert "rmse" in engine.state.metrics
    res = engine.state.metrics['rmse']

    y_preds_full = []
    for i in range(dist.get_world_size()):
        y_preds_full.append((i + 1) * torch.ones(offset))
    y_preds_full = torch.stack(y_preds_full).to(device).flatten()

    true_res = np.sqrt(np.mean(np.square((y_true - y_preds_full).cpu().numpy())))

    assert pytest.approx(res) == true_res


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):

    device = "cuda:{}".format(local_rank)
    _test_distrib_itegration(device)


@pytest.mark.distributed
def test_distrib_cpu(local_rank, distributed_context_single_node_gloo):

    device = "cpu"
    _test_distrib_itegration(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif('MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = "cpu"
    _test_distrib_itegration(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif('GPU_MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = "cuda:{}".format(distributed_context_multi_node_nccl['local_rank'])
    _test_distrib_itegration(device)
