import os
from typing import Tuple
from unittest.mock import patch

import pytest
import sklearn
import torch
from sklearn.metrics import precision_recall_curve

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.metrics.epoch_metric import EpochMetricWarning
from ignite.metrics.precision_recall_curve import PrecisionRecallCurve


@pytest.fixture()
def mock_no_sklearn():
    with patch.dict("sys.modules", {"sklearn.metrics": None}):
        yield sklearn


def test_no_sklearn(mock_no_sklearn):
    with pytest.raises(ModuleNotFoundError, match=r"This module requires scikit-learn to be installed."):
        y = torch.tensor([1, 1])
        pr_curve = PrecisionRecallCurve()
        pr_curve.update((y, y))
        pr_curve.compute()


def test_precision_recall_curve(available_device):
    size = 100
    y_pred = torch.rand(size, 1, dtype=torch.float32, device=available_device)
    y_true = torch.zeros(size, dtype=torch.float32, device=available_device)
    y_true[size // 2 :] = 1.0
    expected_precision, expected_recall, expected_thresholds = precision_recall_curve(
        y_true.cpu().numpy(), y_pred.cpu().numpy()
    )

    precision_recall_curve_metric = PrecisionRecallCurve(device=available_device)
    assert precision_recall_curve_metric._device == torch.device(available_device)

    precision_recall_curve_metric.update((y_pred, y_true))
    precision, recall, thresholds = precision_recall_curve_metric.compute()

    precision = precision.cpu().numpy()
    recall = recall.cpu().numpy()
    thresholds = thresholds.cpu().numpy()

    assert pytest.approx(precision) == expected_precision
    assert pytest.approx(recall) == expected_recall
    assert thresholds == pytest.approx(expected_thresholds, rel=1e-6)


def test_integration_precision_recall_curve_with_output_transform(available_device):
    size = 100
    y_pred = torch.rand(size, 1, dtype=torch.float32, device=available_device)
    y_true = torch.zeros(size, dtype=torch.float32, device=available_device)
    y_true[size // 2 :] = 1.0
    perm = torch.randperm(size)
    y_pred = y_pred[perm]
    y_true = y_true[perm]

    expected_precision, expected_recall, expected_thresholds = precision_recall_curve(
        y_true.cpu().numpy(), y_pred.cpu().numpy()
    )

    batch_size = 10

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = y_true[idx : idx + batch_size]
        y_pred_batch = y_pred[idx : idx + batch_size]
        return idx, y_pred_batch, y_true_batch

    engine = Engine(update_fn)

    precision_recall_curve_metric = PrecisionRecallCurve(
        output_transform=lambda x: (x[1], x[2]), device=available_device
    )
    assert precision_recall_curve_metric._device == torch.device(available_device)
    precision_recall_curve_metric.attach(engine, "precision_recall_curve")

    data = list(range(size // batch_size))
    precision, recall, thresholds = engine.run(data, max_epochs=1).metrics["precision_recall_curve"]
    precision = precision.cpu().numpy()
    recall = recall.cpu().numpy()
    thresholds = thresholds.cpu().numpy()
    assert pytest.approx(precision) == expected_precision
    assert pytest.approx(recall) == expected_recall
    assert thresholds == pytest.approx(expected_thresholds, rel=1e-6)


def test_integration_precision_recall_curve_with_activated_output_transform(available_device):
    size = 100
    y_pred = torch.rand(size, 1, dtype=torch.float32, device=available_device)
    y_true = torch.zeros(size, dtype=torch.float32, device=available_device)
    y_true[size // 2 :] = 1.0
    perm = torch.randperm(size)
    y_pred = y_pred[perm]
    y_true = y_true[perm]

    sigmoid_y_pred = torch.sigmoid(y_pred).cpu().numpy()
    expected_precision, expected_recall, expected_thresholds = precision_recall_curve(
        y_true.cpu().numpy(), sigmoid_y_pred
    )

    batch_size = 10

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = y_true[idx : idx + batch_size]
        y_pred_batch = y_pred[idx : idx + batch_size]
        return idx, y_pred_batch, y_true_batch

    engine = Engine(update_fn)

    precision_recall_curve_metric = PrecisionRecallCurve(
        output_transform=lambda x: (torch.sigmoid(x[1]), x[2]), device=available_device
    )
    assert precision_recall_curve_metric._device == torch.device(available_device)
    precision_recall_curve_metric.attach(engine, "precision_recall_curve")

    data = list(range(size // batch_size))
    precision, recall, thresholds = engine.run(data, max_epochs=1).metrics["precision_recall_curve"]
    precision = precision.cpu().numpy()
    recall = recall.cpu().numpy()
    thresholds = thresholds.cpu().numpy()

    assert pytest.approx(precision) == expected_precision
    assert pytest.approx(recall) == expected_recall
    assert thresholds == pytest.approx(expected_thresholds, rel=1e-6)


def test_check_compute_fn(available_device):
    y_pred = torch.zeros((8, 13))
    y_pred[:, 1] = 1
    y_true = torch.zeros_like(y_pred)
    output = (y_pred, y_true)

    em = PrecisionRecallCurve(check_compute_fn=True, device=available_device)
    assert em._device == torch.device(available_device)

    em.reset()
    with pytest.warns(EpochMetricWarning, match=r"Probably, there can be a problem with `compute_fn`"):
        em.update(output)

    em = PrecisionRecallCurve(check_compute_fn=False, device=available_device)
    assert em._device == torch.device(available_device)
    em.update(output)


def _test_distrib_compute(device):
    rank = idist.get_rank()

    def _test(y_pred, y, batch_size, metric_device):
        metric_device = torch.device(metric_device)
        prc = PrecisionRecallCurve(device=metric_device)

        prc.reset()
        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                prc.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
        else:
            prc.update((y_pred, y))

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y = y.cpu().numpy()
        np_y_pred = y_pred.cpu().numpy()

        res = prc.compute()

        assert isinstance(res, Tuple)
        assert precision_recall_curve(np_y, np_y_pred)[0] == pytest.approx(res[0].cpu().numpy())
        assert precision_recall_curve(np_y, np_y_pred)[1] == pytest.approx(res[1].cpu().numpy())
        assert precision_recall_curve(np_y, np_y_pred)[2] == pytest.approx(res[2].cpu().numpy())

    def get_test_cases():
        test_cases = [
            # Binary input data of shape (N,) or (N, 1)
            (torch.randint(0, 2, size=(10,)), torch.randint(0, 2, size=(10,)), 1),
            (torch.randint(0, 2, size=(10, 1)), torch.randint(0, 2, size=(10, 1)), 1),
            # updated batches
            (torch.randint(0, 2, size=(50,)), torch.randint(0, 2, size=(50,)), 16),
            (torch.randint(0, 2, size=(50, 1)), torch.randint(0, 2, size=(50, 1)), 16),
        ]
        return test_cases

    for i in range(3):
        torch.manual_seed(12 + rank + i)
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            y_pred = y_pred.to(device)
            y = y.to(device)
            _test(y_pred, y, batch_size, "cpu")
            if device.type != "xla":
                _test(y_pred, y, batch_size, idist.device())


def _test_distrib_integration(device):
    rank = idist.get_rank()

    def _test(n_epochs, metric_device):
        metric_device = torch.device(metric_device)
        n_iters = 80
        batch_size = 151
        torch.manual_seed(12 + rank)

        y_true = torch.randint(0, 2, (n_iters * batch_size,)).to(device)
        y_preds = torch.randint(0, 2, (n_iters * batch_size,)).to(device)

        def update(engine, i):
            return (
                y_preds[i * batch_size : (i + 1) * batch_size],
                y_true[i * batch_size : (i + 1) * batch_size],
            )

        engine = Engine(update)

        prc = PrecisionRecallCurve(device=metric_device)
        prc.attach(engine, "prc")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        y_true = idist.all_gather(y_true)
        y_preds = idist.all_gather(y_preds)

        assert "prc" in engine.state.metrics

        precision, recall, thresholds = engine.state.metrics["prc"]

        np_y_true = y_true.cpu().numpy().ravel()
        np_y_preds = y_preds.cpu().numpy().ravel()

        expected_precision, expected_recall, expected_thresholds = precision_recall_curve(np_y_true, np_y_preds)

        assert precision.shape == expected_precision.shape
        assert recall.shape == expected_recall.shape
        assert thresholds.shape == expected_thresholds.shape
        assert pytest.approx(precision.cpu().numpy()) == expected_precision
        assert pytest.approx(recall.cpu().numpy()) == expected_recall
        assert pytest.approx(thresholds.cpu().numpy()) == expected_thresholds

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
