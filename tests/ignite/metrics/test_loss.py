import os

import torch
from torch import nn
from torch.nn.functional import nll_loss

from ignite.exceptions import NotComputableError
from ignite.metrics import Loss

import pytest
from numpy.testing import assert_almost_equal


def test_zero_div():
    loss = Loss(nll_loss)
    with pytest.raises(NotComputableError):
        loss.compute()


def test_compute():
    loss = Loss(nll_loss)

    y_pred = torch.tensor([[0.1, 0.4, 0.5], [0.1, 0.7, 0.2]]).log()
    y = torch.tensor([2, 2]).long()
    loss.update((y_pred, y))
    assert_almost_equal(loss.compute(), 1.1512925625)

    y_pred = torch.tensor([[0.1, 0.3, 0.6], [0.6, 0.2, 0.2], [0.2, 0.7, 0.1]]).log()
    y = torch.tensor([2, 0, 2]).long()
    loss.update((y_pred, y))
    assert_almost_equal(loss.compute(), 1.1253643036)  # average


def test_compute_on_criterion():
    loss = Loss(nn.NLLLoss())

    y_pred = torch.tensor([[0.1, 0.4, 0.5], [0.1, 0.7, 0.2]]).log()
    y = torch.tensor([2, 2]).long()
    loss.update((y_pred, y))
    assert_almost_equal(loss.compute(), 1.1512925625)

    y_pred = torch.tensor([[0.1, 0.3, 0.6], [0.6, 0.2, 0.2], [0.2, 0.7, 0.1]]).log()
    y = torch.tensor([2, 0, 2]).long()
    loss.update((y_pred, y))
    assert_almost_equal(loss.compute(), 1.1253643036)  # average


def test_non_averaging_loss():
    loss = Loss(nn.NLLLoss(reduction='none'))

    y_pred = torch.tensor([[0.1, 0.4, 0.5], [0.1, 0.7, 0.2]]).log()
    y = torch.tensor([2, 2]).long()
    with pytest.raises(ValueError):
        loss.update((y_pred, y))


def test_kwargs_loss():
    loss = Loss(nll_loss)

    y_pred = torch.tensor([[0.1, 0.4, 0.5], [0.1, 0.7, 0.2]]).log()
    y = torch.tensor([2, 2]).long()
    loss.update((y_pred, y, {"weight": torch.tensor([0, 0, 0], dtype=torch.float)}))
    assert_almost_equal(loss.compute(), 0)


def test_reset():
    loss = Loss(nll_loss)

    y_pred = torch.tensor([[0.1, 0.3, 0.6], [0.6, 0.2, 0.2]]).log()
    y = torch.tensor([2, 0]).long()
    loss.update((y_pred, y))
    loss.compute()
    loss.reset()
    with pytest.raises(NotComputableError):
        loss.compute()


def _test_distrib_compute_on_criterion(device):
    import torch.distributed as dist

    def _gather(y):
        output = [torch.zeros_like(y) for i in range(dist.get_world_size())]
        dist.all_gather(output, y)
        y = torch.cat(output, dim=0)
        return y

    criterion = nn.NLLLoss().to(device)
    loss = Loss(criterion, device=device)

    y_pred = torch.tensor([[0.1, 0.4, 0.5], [0.1, 0.7, 0.2]], device=device).log()
    y = torch.tensor([2, 2], device=device).long()
    loss.update((y_pred, y))
    n = loss._num_examples
    assert n == len(y)
    res = loss.compute()
    assert n * dist.get_world_size() == loss._num_examples

    y_pred = _gather(y_pred)
    y = _gather(y)
    true_loss_value = criterion(y_pred, y)
    assert_almost_equal(res, true_loss_value.item())

    loss.reset()
    y_pred = torch.tensor([[0.1, 0.3, 0.6], [0.6, 0.2, 0.2], [0.2, 0.7, 0.1]], device=device).log()
    y = torch.tensor([2, 0, 2], device=device).long()
    loss.update((y_pred, y))
    n = loss._num_examples
    res = loss.compute()
    assert n * dist.get_world_size() == loss._num_examples

    y_pred = _gather(y_pred)
    y = _gather(y)
    true_loss_value = criterion(y_pred, y)
    assert_almost_equal(res, true_loss_value.item())


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):

    device = "cuda:{}".format(local_rank)
    _test_distrib_compute_on_criterion(device)


@pytest.mark.distributed
def test_distrib_cpu(distributed_context_single_node_gloo):

    device = "cpu"
    _test_distrib_compute_on_criterion(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif('MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = "cpu"
    _test_distrib_compute_on_criterion(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif('GPU_MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = "cuda:{}".format(distributed_context_multi_node_nccl['local_rank'])
    _test_distrib_compute_on_criterion(device)
