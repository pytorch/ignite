import os

import numpy as np
import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.metrics.regression import WaveHedgesDistance


def test_wrong_input_shapes():
    m = WaveHedgesDistance()

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4), torch.rand(4, 1)))

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4, 1), torch.rand(4)))


def test_compute(available_device):
    inputs = [torch.randn(4) for _ in range(4)]
    ground_truth = torch.randn(4)

    m = WaveHedgesDistance(device=available_device)
    assert m._device == torch.device(available_device)

    def compute_sum(x):
        return torch.sum(torch.abs(ground_truth - x) / torch.maximum(ground_truth, x))

    total = 0.0
    for x in inputs:
        m.update((x, ground_truth))
        total += compute_sum(x).item()
        assert m.compute() == pytest.approx(total)


@pytest.mark.parametrize("n_times", range(5))
@pytest.mark.parametrize(
    "y_pred, y, batch_size",
    [
        (torch.rand(size=(100,)), torch.rand(size=(100,)), 10),
        (torch.rand(size=(100, 1)), torch.rand(size=(100, 1)), 20),
    ],
)
def test_integration_wave_hedges_distance(n_times, y_pred, y, batch_size, available_device):
    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        return y_pred[idx : idx + batch_size], y[idx : idx + batch_size]

    engine = Engine(update_fn)

    m = WaveHedgesDistance(device=available_device)
    assert m._device == torch.device(available_device)
    m.attach(engine, "whd")

    data = list(range(y_pred.shape[0] // batch_size))
    whd = engine.run(data, max_epochs=1).metrics["whd"]

    flat_pred = y_pred.view(-1).cpu()
    flat_true = y.view(-1).cpu()
    expected = torch.sum(torch.abs(flat_true - flat_pred) / torch.maximum(flat_true, flat_pred))

    assert whd == pytest.approx(expected.item())


def _test_distrib_compute(device):
    rank = idist.get_rank()

    def _test(metric_device):
        metric_device = torch.device(metric_device)
        m = WaveHedgesDistance(device=metric_device)

        y_pred = torch.randint(0, 10, size=(10,), device=device).float()
        y = torch.randint(0, 10, size=(10,), device=device).float()

        m.update((y_pred, y))

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y_pred = y_pred.cpu().numpy()
        np_y = y.cpu().numpy()

        res = m.compute()

        np_sum = (np.abs(np_y - np_y_pred) / (np.maximum.reduce([np_y_pred, np_y]) + 1e-30)).sum()

        assert np_sum == pytest.approx(res)

    for i in range(3):
        torch.manual_seed(10 + rank + i)
        _test("cpu")
        if device.type != "xla":
            _test(idist.device())


def _test_distrib_integration(device):
    rank = idist.get_rank()

    def _test(n_epochs, metric_device):
        metric_device = torch.device(metric_device)
        n_iters = 80
        batch_size = 16

        y_true = torch.rand(size=(n_iters * batch_size,)).to(device)
        y_preds = torch.rand(size=(n_iters * batch_size,)).to(device)

        def update(engine, i):
            return (
                y_preds[i * batch_size : (i + 1) * batch_size],
                y_true[i * batch_size : (i + 1) * batch_size],
            )

        engine = Engine(update)

        m = WaveHedgesDistance(device=metric_device)
        m.attach(engine, "whm")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        y_preds = idist.all_gather(y_preds)
        y_true = idist.all_gather(y_true)

        assert "whm" in engine.state.metrics

        res = engine.state.metrics["whm"]

        np_y_true = y_true.cpu().numpy()
        np_y_preds = y_preds.cpu().numpy()

        np_sum = (np.abs(np_y_true - np_y_preds) / (np.maximum.reduce([np_y_preds, np_y_true]) + 1e-30)).sum()

        assert pytest.approx(res) == np_sum

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        for i in range(2):
            torch.manual_seed(12 + rank + i)
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
