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


def test_geometric_mean_relative_absolute_error():
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


def test_geometric_mean_relative_absolute_error_2():

    np.random.seed(1)
    size = 105
    np_y_pred = np.random.rand(size, 1)
    np_y = np.random.rand(size, 1)
    np.random.shuffle(np_y)

    np_y_sum = 0
    num_examples = 0
    num_sum_of_errors = 0
    np_gmrae = 0

    m = GeometricMeanRelativeAbsoluteError()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    m.reset()
    n_iters = 15
    batch_size = size // n_iters
    for i in range(n_iters + 1):
        idx = i * batch_size
        np_y_i = np_y[idx : idx + batch_size]
        np_y_pred_i = np_y_pred[idx : idx + batch_size]

        np_y_sum += np_y_i.sum()
        num_examples += np_y_i.shape[0]
        np_mean = np_y_sum / num_examples

        np_gmrae += np.log(np.abs(np_y_i - np_y_pred_i) / np.abs(np_y_i - np_mean)).sum()
        m.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

    assert np.exp(np_gmrae / num_examples) == pytest.approx(m.compute())


def test_integration_geometric_mean_relative_absolute_error():

    np.random.seed(1)
    size = 105
    np_y_pred = np.random.rand(size, 1)
    np_y = np.random.rand(size, 1)
    np.random.shuffle(np_y)

    np_y_sum = 0
    num_examples = 0
    num_sum_of_errors = 0
    np_gmrae = 0

    n_iters = 15
    batch_size = size // n_iters
    for i in range(n_iters + 1):
        idx = i * batch_size
        np_y_i = np_y[idx : idx + batch_size]
        np_y_pred_i = np_y_pred[idx : idx + batch_size]

        np_y_sum += np_y_i.sum()
        num_examples += np_y_i.shape[0]
        np_mean = np_y_sum / num_examples

        np_gmrae += np.log(np.abs(np_y_i - np_y_pred_i) / np.abs(np_y_i - np_mean)).sum()

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]
        return torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    m = GeometricMeanRelativeAbsoluteError()
    m.attach(engine, "geometric_mean_relative_absolute_error")

    data = list(range(size // batch_size))
    gmrae = engine.run(data, max_epochs=1).metrics["geometric_mean_relative_absolute_error"]

    assert np.exp(np_gmrae / num_examples) == pytest.approx(m.compute())


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

        # sum_y = 0
        # num_examples = 0
        # sum_of_errors = 0
        # np_gmrae = 0
        # sum_y += np_y.sum()
        # num_examples += np_y.shape[0]
        # y_mean = sum_y / num_examples
        # numerator = np.abs(y.view_as(y_pred) - y_pred)
        # denominator = np.abs(y.view_as(y_pred) - y_mean)
        # sum_of_errors += np.log(numerator / denominator).sum()
        # np_gmrae += np.exp((sum_of_errors / num_examples).mean())

        assert m.compute() == pytest.approx(np_gmrae)

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

        np_y_sum = 0
        num_examples = 0
        np_gmrae = 0
        for i in range(n_iters + 1):
            np_y_i = np_y[i * s + rank * offset : (i + 1) * s + rank * offset]
            np_y_pred_i = np_y_pred[i * s + rank * offset : (i + 1) * s + rank * offset]

            np_y_sum += np_y_i.sum()
            num_examples += np_y_i.shape[0]
            np_mean = np_y_sum / num_examples

            np_gmrae += np.log(np.abs(np_y_i - np_y_pred_i) / np.abs(np_y_i - np_mean)).sum()

        assert pytest.approx(res, rel=1e-4) == np.exp(np_gmrae / num_examples)

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
