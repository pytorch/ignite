import os

import numpy as np
import pytest
import torch

import ignite.distributed as idist
from ignite.contrib.metrics.regression import GeometricMeanRelativeAbsoluteError
from ignite.engine import Engine
from ignite.exceptions import NotComputableError


def test_zero_sample():
    m = GeometricMeanRelativeAbsoluteError()
    with pytest.raises(
        NotComputableError,
        match=r"GeometricMeanRelativeAbsoluteError must have at least one example before it can be computed",
    ):
        m.compute()


def test_wrong_input_shapes():
    m = GeometricMeanRelativeAbsoluteError()

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4), torch.rand(4, 1)))

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4, 1), torch.rand(4,)))


def test_compute():
    size = 51
    np_y_pred = np.random.rand(size,)
    np_y = np.random.rand(size,)
    np_gmrae = np.exp(np.log(np.abs(np_y - np_y_pred) / np.abs(np_y - np_y.mean())).mean())

    m = GeometricMeanRelativeAbsoluteError()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    m.reset()
    m.update((y_pred, y))

    assert np_gmrae == pytest.approx(m.compute())


def test_integration():

    y_pred = torch.rand(size=(100,))
    y = torch.rand(size=(100,))

    batch_size = 10

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]
        return torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    m = GeometricMeanRelativeAbsoluteError()
    m.attach(engine, "gmrae")

    np_y = y.numpy().ravel()
    np_y_pred = y_pred.numpy().ravel()

    data = list(range(y_pred.shape[0] // batch_size))
    gmrae = engine.run(data, max_epochs=1).metrics["gmrae"]

    sum_errors = np.log(np.abs(np_y - np_y_pred) / np.abs(np_y - np_y.mean())).sum()
    np_len = len(y_pred)
    np_ans = np.exp(sum_errors / np_len)

    assert np_ans == pytest.approx(gmrae)


def _test_distrib_compute(device):

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(metric_device):
        metric_device = torch.device(metric_device)
        m = GeometricMeanRelativeAbsoluteError(device=metric_device)
        torch.manual_seed(10 + rank)

        y_pred = torch.rand(size=(100,), device=device)
        y = torch.rand(size=(100,), device=device)

        m.update((y_pred, y))

        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y = y.cpu().numpy()
        np_y_pred = y_pred.cpu().numpy()

        np_gmrae = np.exp(np.log(np.abs(np_y - np_y_pred) / np.abs(np_y - np_y.mean())).mean())

        assert m.compute() == pytest.approx(np_gmrae, rel=1e-4)

    for _ in range(3):
        _test("cpu")
        if device.type != "xla":
            _test(idist.device())


def _test_distrib_integration(device):

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(n_epochs, metric_device):
        metric_device = torch.device(metric_device)
        n_iters = 80
        s = 16
        n_classes = 2

        offset = n_iters * s
        y_true = torch.rand(size=(offset * idist.get_world_size(),)).to(device)
        y_preds = torch.rand(size=(offset * idist.get_world_size(),)).to(device)

        def update(engine, i):
            return (
                y_preds[i * s + rank * offset : (i + 1) * s + rank * offset],
                y_true[i * s + rank * offset : (i + 1) * s + rank * offset],
            )

        engine = Engine(update)

        gmrae = GeometricMeanRelativeAbsoluteError(device=metric_device)
        gmrae.attach(engine, "gmrae")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        assert "gmrae" in engine.state.metrics

        res = engine.state.metrics["gmrae"]

        np_y = y_true.cpu().numpy()
        np_y_pred = y_preds.cpu().numpy()

        np_gmrae = np.exp(np.log(np.abs(np_y - np_y_pred) / np.abs(np_y - np_y.mean())).mean())

        assert pytest.approx(res, rel=1e-4) == np_gmrae

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        for _ in range(2):
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
