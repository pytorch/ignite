import os

import numpy as np
import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.regression import MedianAbsoluteError


def test_zero_sample():
    m = MedianAbsoluteError()
    with pytest.raises(
        NotComputableError, match=r"EpochMetric must have at least one example before it can be computed"
    ):
        m.compute()


def test_wrong_input_shapes():
    m = MedianAbsoluteError()

    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        m.update((torch.rand(4, 1, 2), torch.rand(4, 1)))

    with pytest.raises(ValueError, match=r"Targets should be of shape"):
        m.update((torch.rand(4, 1), torch.rand(4, 1, 2)))

    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        m.update((torch.rand(4, 1, 2), torch.rand(4)))

    with pytest.raises(ValueError, match=r"Targets should be of shape"):
        m.update((torch.rand(4), torch.rand(4, 1, 2)))


def test_median_absolute_error(available_device):
    # See https://github.com/torch/torch7/pull/182
    # For even number of elements, PyTorch returns middle element
    # NumPy returns average of middle elements
    # Size of dataset will be odd for these tests

    size = 51
    y_pred = torch.rand(size)
    y = torch.rand(size)
    expected_median_absolute_error = torch.median(torch.abs((y - y_pred).cpu())).item()

    m = MedianAbsoluteError(device=available_device)
    assert m._device == torch.device(available_device)

    m.reset()
    m.update((y_pred, y))

    assert expected_median_absolute_error == pytest.approx(m.compute())


def test_median_absolute_error_2(available_device):
    size = 105
    y_pred = torch.rand(size, 1)
    y = torch.rand(size, 1)
    y = y[torch.randperm(size)]
    expected_median_absolute_error = torch.median(torch.abs(y.cpu() - y_pred.cpu())).item()

    m = MedianAbsoluteError(device=available_device)
    assert m._device == torch.device(available_device)

    m.reset()
    batch_size = 16
    n_iters = size // batch_size + 1
    for i in range(n_iters):
        idx = i * batch_size
        m.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

    assert expected_median_absolute_error == pytest.approx(m.compute())


def test_integration_median_absolute_error(available_device):
    size = 105
    y_pred = torch.rand(size, 1)
    y = torch.rand(size, 1)
    y = y[torch.randperm(size)]

    expected = torch.median(torch.abs(y.cpu() - y_pred.cpu())).item()
    batch_size = 15

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        return y_pred[idx : idx + batch_size], y[idx : idx + batch_size]

    engine = Engine(update_fn)

    m = MedianAbsoluteError(device=available_device)
    assert m._device == torch.device(available_device)
    m.attach(engine, "median_absolute_error")

    data = list(range(size // batch_size))
    median_absolute_error = engine.run(data, max_epochs=1).metrics["median_absolute_error"]

    assert expected == pytest.approx(median_absolute_error)


def _test_distrib_compute(device):
    def _test(metric_device):
        metric_device = torch.device(metric_device)
        m = MedianAbsoluteError(device=metric_device)

        size = 105

        y_pred = torch.randint(1, 10, size=(size, 1), dtype=torch.double, device=device)
        y = torch.randint(1, 10, size=(size, 1), dtype=torch.double, device=device)
        m.update((y_pred, y))

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y_pred = y_pred.cpu().numpy().ravel()
        np_y = y.cpu().numpy().ravel()

        res = m.compute()

        e = np.abs(np_y - np_y_pred)

        np_res = np.median(e)
        assert pytest.approx(res) == np_res

    rank = idist.get_rank()
    for i in range(3):
        torch.manual_seed(10 + rank + i)
        _test("cpu")
        if device.type != "xla":
            _test(idist.device())


def _test_distrib_integration(device):
    def _test(n_epochs, metric_device):
        metric_device = torch.device(metric_device)
        n_iters = 80
        batch_size = 105
        y_true = torch.rand(size=(n_iters * batch_size,)).to(device)
        y_preds = torch.rand(size=(n_iters * batch_size,)).to(device)

        def update(engine, i):
            return (
                y_preds[i * batch_size : (i + 1) * batch_size],
                y_true[i * batch_size : (i + 1) * batch_size],
            )

        engine = Engine(update)

        m = MedianAbsoluteError(device=metric_device)
        m.attach(engine, "mae")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        y_preds = idist.all_gather(y_preds)
        y_true = idist.all_gather(y_true)

        assert "mae" in engine.state.metrics

        res = engine.state.metrics["mae"]

        np_y_true = y_true.cpu().numpy().ravel()
        np_y_preds = y_preds.cpu().numpy().ravel()

        e = np.abs(np_y_true - np_y_preds)
        np_res = np.median(e)

        assert pytest.approx(res) == np_res

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        rank = idist.get_rank()
        for i in range(2):
            torch.manual_seed(10 + rank + i)
            _test(n_epochs=1, metric_device=metric_device)
            _test(n_epochs=2, metric_device=metric_device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib_compute(device)
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib_compute(device)
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_compute, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_integration, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):
    device = idist.device()
    _test_distrib_compute(device)
    _test_distrib_integration(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):
    device = idist.device()
    _test_distrib_compute(device)
    _test_distrib_integration(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_compute(device)
    _test_distrib_integration(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_compute(device)
    _test_distrib_integration(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)
