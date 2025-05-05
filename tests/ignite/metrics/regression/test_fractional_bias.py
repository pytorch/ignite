import os

import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.regression import FractionalBias


def test_zero_sample():
    m = FractionalBias()
    with pytest.raises(
        NotComputableError, match=r"FractionalBias must have at least one example before it can be computed"
    ):
        m.compute()


def test_wrong_input_shapes():
    m = FractionalBias()

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4), torch.rand(4, 1)))

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4, 1), torch.rand(4)))


def test_fractional_bias(available_device):
    a = torch.randn(4)
    b = torch.randn(4)
    c = torch.randn(4)
    d = torch.randn(4)
    ground_truth = torch.randn(4)

    m = FractionalBias(device=available_device)
    assert m._device == torch.device(available_device)

    total_error = 0.0
    total_len = 0

    for pred in [a, b, c, d]:
        m.update((pred, ground_truth))

        error = 2 * (ground_truth - pred) / (pred + ground_truth)
        total_error += error.sum().item()
        total_len += len(pred)

        expected = total_error / total_len
        assert m.compute() == pytest.approx(expected)


@pytest.mark.parametrize("n_times", range(5))
@pytest.mark.parametrize(
    "test_case",
    [
        (torch.rand(size=(100,)), torch.rand(size=(100,)), 10),
        (torch.rand(size=(100, 1)), torch.rand(size=(100, 1)), 20),
    ],
)
def test_integration_fractional_bias(n_times, test_case, available_device):
    y_pred, y, batch_size = test_case

    np_y = y.double().numpy().ravel()
    np_y_pred = y_pred.double().numpy().ravel()

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]

        torch_y_pred_batch = (
            torch.from_numpy(y_pred_batch).to(dtype=torch.float32)
            if available_device == "mps"
            else torch.from_numpy(y_pred_batch)
        )
        torch_y_true_batch = (
            torch.from_numpy(y_true_batch).to(dtype=torch.float32)
            if available_device == "mps"
            else torch.from_numpy(y_true_batch)
        )

        return torch_y_pred_batch, torch_y_true_batch

    engine = Engine(update_fn)

    metric = FractionalBias(device=available_device)
    assert metric._device == torch.device(available_device)

    metric.attach(engine, "fb")

    data = list(range(y_pred.shape[0] // batch_size))
    fb = engine.run(data, max_epochs=1).metrics["fb"]

    expected = (2 * (np_y - np_y_pred) / (np_y_pred + np_y)).sum() / len(np_y)

    if available_device == "mps":
        assert expected == pytest.approx(fb, rel=1e-5)
    else:
        assert expected == pytest.approx(fb)


def test_error_is_not_nan(available_device):
    m = FractionalBias(device=available_device)
    assert m._device == torch.device(available_device)
    m.update((torch.zeros(4), torch.zeros(4)))
    assert not (torch.isnan(m._sum_of_errors).any() or torch.isinf(m._sum_of_errors).any()), m._sum_of_errors


def _test_distrib_compute(device, tol=1e-5):
    rank = idist.get_rank()

    def _test(metric_device):
        metric_device = torch.device(metric_device)
        m = FractionalBias(device=metric_device)

        y_pred = torch.randint(0, 10, size=(10,), device=device).float()
        y = torch.randint(0, 10, size=(10,), device=device).float()

        m.update((y_pred, y))

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y_pred = y_pred.cpu().numpy()
        np_y = y.cpu().numpy()

        res = m.compute()

        np_sum = (2 * (np_y - np_y_pred) / (np_y_pred + np_y + 1e-30)).sum()
        np_len = len(y_pred)
        np_ans = np_sum / np_len

        assert np_ans == pytest.approx(res, rel=tol)

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

        y_true = torch.rand(size=(n_iters * batch_size,), dtype=torch.double).to(device)
        y_preds = torch.rand(size=(n_iters * batch_size,), dtype=torch.double).to(device)

        def update(engine, i):
            return (
                y_preds[i * batch_size : (i + 1) * batch_size],
                y_true[i * batch_size : (i + 1) * batch_size],
            )

        engine = Engine(update)

        m = FractionalBias(device=metric_device)
        m.attach(engine, "fb")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        y_preds = idist.all_gather(y_preds)
        y_true = idist.all_gather(y_true)

        assert "fb" in engine.state.metrics

        res = engine.state.metrics["fb"]
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()

        np_y_true = y_true.cpu().numpy()
        np_y_preds = y_preds.cpu().numpy()

        np_sum = (2 * (np_y_true - np_y_preds) / (np_y_preds + np_y_true + 1e-30)).sum()
        np_len = len(y_preds)
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
    _test_distrib_integration(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_compute(device, tol=1e-4)
    _test_distrib_integration(device, tol=1e-4)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_compute(device, tol=1e-4)
    _test_distrib_integration(device, tol=1e-4)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)
