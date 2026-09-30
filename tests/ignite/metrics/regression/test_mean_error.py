import os

import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.regression import MeanError


def test_zero_sample():
    m = MeanError()
    with pytest.raises(NotComputableError, match=r"MeanError must have at least one example before it can be computed"):
        m.compute()


def test_wrong_input_shapes():
    m = MeanError()

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4), torch.rand(4, 1)))

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4, 1), torch.rand(4)))


def test_mean_error(available_device):
    preds = [torch.randn(4) for _ in range(4)]
    ground_truth = torch.randn(4)

    m = MeanError(device=available_device)
    assert m._device == torch.device(available_device)

    total_error = 0.0
    total_count = 0

    for pred in preds:
        m.update((pred, ground_truth))
        total_error += torch.sum(ground_truth - pred).item()
        total_count += pred.numel()
        expected = total_error / total_count
        assert m.compute() == pytest.approx(expected)


@pytest.mark.parametrize("n_times", range(5))
@pytest.mark.parametrize(
    "y_pred, y, batch_size",
    [
        (torch.rand((50,)), torch.rand((50,)), 1),
        (torch.rand((50, 1)), torch.rand((50, 1)), 10),
    ],
)
def test_integration_mean_error(n_times, y_pred, y, batch_size, available_device):
    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        return y_pred[idx : idx + batch_size], y[idx : idx + batch_size]

    engine = Engine(update_fn)

    metric = MeanError(device=available_device)
    assert metric._device == torch.device(available_device)
    metric.attach(engine, "me")

    data = list(range(y_pred.shape[0] // batch_size))
    result = engine.run(data, max_epochs=1).metrics["me"]

    expected = torch.mean(y - y_pred).item()
    assert result == pytest.approx(expected, rel=1e-4)


def _test_distrib_compute(device):
    rank = idist.get_rank()

    def _test(metric_device):
        metric_device = torch.device(metric_device)
        m = MeanError(device=metric_device)

        y_pred = torch.rand(size=(100,), device=device)
        y = torch.rand(size=(100,), device=device)

        m.update((y_pred, y))

        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y = y.cpu().numpy()
        np_y_pred = y_pred.cpu().numpy()

        np_sum = (np_y - np_y_pred).sum()
        np_len = len(np_y_pred)
        np_ans = np_sum / np_len

        assert m.compute() == pytest.approx(np_ans, rel=1e-5)

    for i in range(3):
        torch.manual_seed(10 + rank + i)
        _test("cpu")
        if device.type != "xla":
            _test(idist.device())


def _test_distrib_integration(device, tol=1e-5):
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

        me = MeanError(device=metric_device)
        me.attach(engine, "me")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        y_preds = idist.all_gather(y_preds)
        y_true = idist.all_gather(y_true)

        assert "me" in engine.state.metrics

        res = engine.state.metrics["me"]

        np_y = y_true.cpu().numpy()
        np_y_pred = y_preds.cpu().numpy()

        np_sum = (np_y - np_y_pred).sum()
        np_len = len(np_y_pred)
        np_ans = np_sum / np_len

        assert pytest.approx(res, rel=tol) == np_ans

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
