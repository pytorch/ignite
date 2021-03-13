import os

import pytest
import torch
from numpy.testing import assert_almost_equal
from torch import nn
from torch.nn.functional import nll_loss

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics import Loss


def y_test_1(requires_grad=False, device=None):
    return (
        torch.tensor([[0.1, 0.4, 0.5], [0.1, 0.7, 0.2]], device=device, requires_grad=requires_grad).log(),
        torch.tensor([2, 2], device=device).long(),
        1.1512925625,
    )


def y_test_2():
    return (
        torch.tensor([[0.1, 0.3, 0.6], [0.6, 0.2, 0.2], [0.2, 0.7, 0.1]]).log(),
        torch.tensor([2, 0, 2]).long(),
        1.1253643036,
    )


def y_test_3():
    return torch.tensor([[0.1, 0.3, 0.6], [0.6, 0.2, 0.2]]).log(), torch.tensor([2, 0]).long()


def test_zero_div():
    loss = Loss(nll_loss)
    with pytest.raises(NotComputableError, match=r"Loss must have at least one example before it can be computed"):
        loss.compute()


def test_compute():
    def _test(y_test_1, y_test_2):
        loss = Loss(nll_loss)

        y_pred, y, expected_loss = y_test_1
        loss.update((y_pred, y))
        assert_almost_equal(loss.compute(), expected_loss)

        y_pred, y, expected_loss = y_test_2
        loss.update((y_pred, y))
        assert_almost_equal(loss.compute(), expected_loss)  # average

    _test(y_test_1(), y_test_2())


def test_compute_on_criterion():
    def _test(y_test_1, y_test_2):
        loss = Loss(nn.NLLLoss())

        y_pred, y, expected_loss = y_test_1
        loss.update((y_pred, y))
        assert_almost_equal(loss.compute(), expected_loss)

        y_pred, y, expected_loss = y_test_2
        loss.update((y_pred, y))
        assert_almost_equal(loss.compute(), expected_loss)  # average

    _test(y_test_1(), y_test_2())


def test_non_averaging_loss():
    loss = Loss(nn.NLLLoss(reduction="none"))

    y_pred, y, _ = y_test_1()
    with pytest.raises(ValueError):
        loss.update((y_pred, y))


def test_gradient_based_loss():
    # Tests https://github.com/pytorch/ignite/issues/1674
    x = torch.tensor([[0.1, 0.4, 0.5], [0.1, 0.7, 0.2]], requires_grad=True)
    y_pred = x.mm(torch.randn(size=(3, 1)))

    def loss_fn(y_pred, x):
        gradients = torch.autograd.grad(
            outputs=y_pred, inputs=x, grad_outputs=torch.ones_like(y_pred), create_graph=True,
        )[0]

        gradients = gradients.flatten(start_dim=1)

        return gradients.norm(2, dim=1).mean()

    loss = Loss(loss_fn)
    loss.update((y_pred, x))


def test_kwargs_loss():
    loss = Loss(nll_loss)

    y_pred, y, _ = y_test_1()
    loss.update((y_pred, y, {"weight": torch.tensor([0, 0, 0], dtype=torch.float)}))
    assert_almost_equal(loss.compute(), 0)


def test_reset():
    loss = Loss(nll_loss)

    y_pred, y = y_test_3()
    loss.update((y_pred, y))
    loss.compute()
    loss.reset()
    with pytest.raises(NotComputableError):
        loss.compute()


def _test_distrib_compute_on_criterion(device, y_test_1, y_test_2, tol=None):
    def _test(metric_device, y_test_1, y_test_2):
        criterion = nn.NLLLoss().to(device)
        loss = Loss(criterion, device=metric_device)

        y_pred, y, _ = y_test_1
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
        y_pred, y, _ = y_test_2
        loss.update((y_pred, y))
        n = loss._num_examples
        res = loss.compute()
        assert n * idist.get_world_size() == loss._num_examples

        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)
        true_loss_value = criterion(y_pred, y)
        if tol is None:
            assert_almost_equal(res, true_loss_value.item())
        else:
            assert pytest.approx(res, rel=tol) == true_loss_value.item()

    _test("cpu", y_test_1, y_test_2)
    if device.type != "xla":
        _test(idist.device(), y_test_1, y_test_2)


def _test_distrib_accumulator_device(device, y_test_1):

    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        loss = Loss(nll_loss, device=metric_device)
        assert loss._device == metric_device
        assert (
            loss._sum.device == metric_device
        ), f"{type(loss._sum.device)}:{loss._sum.device} vs {type(metric_device)}:{metric_device}"

        y_pred, y, _ = y_test_1
        loss.update((y_pred, y))

        assert (
            loss._sum.device == metric_device
        ), f"{type(loss._sum.device)}:{loss._sum.device} vs {type(metric_device)}:{metric_device}"


def test_sum_detached():
    loss = Loss(nll_loss)

    y_pred, y, _ = y_test_1(requires_grad=True)
    loss.update((y_pred, y))

    assert not loss._sum.requires_grad


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):

    device = torch.device(f"cuda:{local_rank}")
    _test_distrib_compute_on_criterion(device, y_test_1(), y_test_2())
    _test_distrib_accumulator_device(device, y_test_1())


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(distributed_context_single_node_gloo):

    device = torch.device("cpu")
    _test_distrib_compute_on_criterion(device, y_test_1(), y_test_2())
    _test_distrib_accumulator_device(device, y_test_1())


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_compute_on_criterion, (device, y_test_1(), y_test_2()), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_accumulator_device, (device, y_test_1()), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = torch.device("cpu")
    _test_distrib_compute_on_criterion(device, y_test_1(), y_test_2(), tol=1e-6)
    _test_distrib_accumulator_device(device, y_test_1())


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = torch.device(f"cuda:{distributed_context_multi_node_nccl['local_rank']}")
    _test_distrib_compute_on_criterion(device, y_test_1(), y_test_2())
    _test_distrib_accumulator_device(device, y_test_1())


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_compute_on_criterion(device, y_test_1(), y_test_2())
    _test_distrib_accumulator_device(device, y_test_1())


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_compute_on_criterion(device, y_test_1(), y_test_2())
    _test_distrib_accumulator_device(device, y_test_1())


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)
