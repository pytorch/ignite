import os

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import MeanPairwiseDistance

import pytest
from pytest import approx


def test_zero_div():
    mpd = MeanPairwiseDistance()
    with pytest.raises(NotComputableError):
        mpd.compute()


def test_compute():
    mpd = MeanPairwiseDistance()

    y_pred = torch.Tensor([[3.0, 4.0], [-3.0, -4.0]])
    y = torch.zeros(2, 2)
    mpd.update((y_pred, y))
    assert isinstance(mpd.compute(), float)
    assert mpd.compute() == approx(5.0)

    mpd.reset()
    y_pred = torch.Tensor([[4.0, 4.0, 4.0, 4.0], [-4.0, -4.0, -4.0, -4.0]])
    y = torch.zeros(2, 4)
    mpd.update((y_pred, y))
    assert isinstance(mpd.compute(), float)
    assert mpd.compute() == approx(8.0)


def _test_distrib_itegration(device):
    import numpy as np
    import torch.distributed as dist
    from ignite.engine import Engine

    rank = dist.get_rank()
    torch.manual_seed(12)

    n_iters = 100
    s = 50
    offset = n_iters * s

    y_true = torch.rand(offset * dist.get_world_size(), 10).to(device)
    y_preds = torch.rand(offset * dist.get_world_size(), 10).to(device)

    def update(engine, i):
        return y_preds[i * s + offset * rank:(i + 1) * s + offset * rank, ...], \
            y_true[i * s + offset * rank:(i + 1) * s + offset * rank, ...]

    engine = Engine(update)

    m = MeanPairwiseDistance(device=device)
    m.attach(engine, "mpwd")

    data = list(range(n_iters))
    engine.run(data=data, max_epochs=1)

    assert "mpwd" in engine.state.metrics
    res = engine.state.metrics['mpwd']

    true_res = []
    for i in range(n_iters * dist.get_world_size()):
        true_res.append(
            torch.pairwise_distance(y_true[i * s:(i + 1) * s, ...],
                                    y_preds[i * s:(i + 1) * s, ...],
                                    p=m._p, eps=m._eps).cpu().numpy()
        )
    true_res = np.array(true_res).ravel()
    true_res = true_res.mean()

    assert pytest.approx(res) == true_res


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):
    device = "cuda:{}".format(local_rank)
    _test_distrib_itegration(device)


@pytest.mark.distributed
def test_distrib_cpu(distributed_context_single_node_gloo):
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
