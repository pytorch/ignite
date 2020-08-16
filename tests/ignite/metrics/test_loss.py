import os

import pytest
import torch
from numpy.testing import assert_almost_equal
from torch import nn
from torch.nn.functional import nll_loss

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics import Loss


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
    loss = Loss(nn.NLLLoss(reduction="none"))

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

    criterion = nn.NLLLoss().to(device)
    loss = Loss(criterion, device=device)

    y_pred = torch.tensor([[0.1, 0.4, 0.5], [0.1, 0.7, 0.2]], device=device).log()
    y = torch.tensor([2, 2], device=device).long()
    loss.update((y_pred, y))
    n = loss._num_examples
    assert n == len(y)
    res = loss.compute()
    assert n * idist.get_world_size() == loss._num_examples

    y_pred = idist.all_gather(y_pred)
    y = idist.all_gather(y)
    true_loss_value = criterion(y_pred, y)
    assert_almost_equal(res, true_loss_value.item())

    loss.reset()
    y_pred = torch.tensor([[0.1, 0.3, 0.6], [0.6, 0.2, 0.2], [0.2, 0.7, 0.1]], device=device).log()
    y = torch.tensor([2, 0, 2], device=device).long()
    loss.update((y_pred, y))
    n = loss._num_examples
    res = loss.compute()
    assert n * idist.get_world_size() == loss._num_examples

    y_pred = idist.all_gather(y_pred)
    y = idist.all_gather(y)
    true_loss_value = criterion(y_pred, y)
    assert_almost_equal(res, true_loss_value.item())


def _test_distrib_sum_device(device):
    device = torch.device(device)
    loss = Loss(nll_loss, device=device)
    assert loss._device == device

    y_pred = torch.tensor([[0.1, 0.4, 0.5], [0.1, 0.7, 0.2]]).log()
    y = torch.tensor([2, 2]).long()
    loss.update((y_pred, y))

    assert loss._sum.device == device


def test_sum_detached():
    loss = Loss(nll_loss)

    y_pred = torch.tensor([[0.1, 0.4, 0.5], [0.1, 0.7, 0.2]], requires_grad=True).log()
    y = torch.tensor([2, 2]).long()
    loss.update((y_pred, y))

    assert not loss._sum.requires_grad


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):

    device = "cuda:{}".format(local_rank)
    _test_distrib_compute_on_criterion(device)
    _test_distrib_sum_device(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(distributed_context_single_node_gloo):

    device = "cpu"
    _test_distrib_compute_on_criterion(device)
    _test_distrib_sum_device(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):

    device = "cpu" if not torch.cuda.is_available() else "cuda"
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_compute_on_criterion, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_sum_device, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = "cpu"
    _test_distrib_compute_on_criterion(device)
    _test_distrib_sum_device(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = "cuda:{}".format(distributed_context_multi_node_nccl["local_rank"])
    _test_distrib_compute_on_criterion(device)
    _test_distrib_sum_device(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_compute_on_criterion(device)
    _test_distrib_sum_device(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_compute_on_criterion(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)
