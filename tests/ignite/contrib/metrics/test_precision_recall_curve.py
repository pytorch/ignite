from typing import Tuple
from unittest.mock import patch

import os
import numpy as np
import pytest
import sklearn
import torch
from sklearn.metrics import precision_recall_curve

import ignite.distributed as idist
from ignite.contrib.metrics.precision_recall_curve import PrecisionRecallCurve
from ignite.engine import Engine
from ignite.metrics.epoch_metric import EpochMetricWarning


@pytest.fixture()
def mock_no_sklearn():
    with patch.dict("sys.modules", {"sklearn.metrics": None}):
        yield sklearn


def test_no_sklearn(mock_no_sklearn):
    with pytest.raises(RuntimeError, match=r"This contrib module requires sklearn to be installed."):
        y = torch.tensor([1, 1])
        pr_curve = PrecisionRecallCurve()
        pr_curve.update((y, y))
        pr_curve.compute()


def test_precision_recall_curve():
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y = np.zeros((size,))
    np_y[size // 2 :] = 1
    sk_precision, sk_recall, sk_thresholds = precision_recall_curve(np_y, np_y_pred)

    precision_recall_curve_metric = PrecisionRecallCurve()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    precision_recall_curve_metric.update((y_pred, y))
    precision, recall, thresholds = precision_recall_curve_metric.compute()

    assert np.array_equal(precision, sk_precision)
    assert np.array_equal(recall, sk_recall)
    # assert thresholds almost equal, due to numpy->torch->numpy conversion
    np.testing.assert_array_almost_equal(thresholds, sk_thresholds)


def test_integration_precision_recall_curve_with_output_transform():
    np.random.seed(1)
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y = np.zeros((size,))
    np_y[size // 2 :] = 1
    np.random.shuffle(np_y)

    sk_precision, sk_recall, sk_thresholds = precision_recall_curve(np_y, np_y_pred)

    batch_size = 10

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    precision_recall_curve_metric = PrecisionRecallCurve(output_transform=lambda x: (x[1], x[2]))
    precision_recall_curve_metric.attach(engine, "precision_recall_curve")

    data = list(range(size // batch_size))
    precision, recall, thresholds = engine.run(data, max_epochs=1).metrics["precision_recall_curve"]

    assert np.array_equal(precision, sk_precision)
    assert np.array_equal(recall, sk_recall)
    # assert thresholds almost equal, due to numpy->torch->numpy conversion
    np.testing.assert_array_almost_equal(thresholds, sk_thresholds)


def test_integration_precision_recall_curve_with_activated_output_transform():
    np.random.seed(1)
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y_pred_sigmoid = torch.sigmoid(torch.from_numpy(np_y_pred)).numpy()
    np_y = np.zeros((size,))
    np_y[size // 2 :] = 1
    np.random.shuffle(np_y)

    sk_precision, sk_recall, sk_thresholds = precision_recall_curve(np_y, np_y_pred_sigmoid)

    batch_size = 10

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    precision_recall_curve_metric = PrecisionRecallCurve(output_transform=lambda x: (torch.sigmoid(x[1]), x[2]))
    precision_recall_curve_metric.attach(engine, "precision_recall_curve")

    data = list(range(size // batch_size))
    precision, recall, thresholds = engine.run(data, max_epochs=1).metrics["precision_recall_curve"]

    assert np.array_equal(precision, sk_precision)
    assert np.array_equal(recall, sk_recall)
    # assert thresholds almost equal, due to numpy->torch->numpy conversion
    np.testing.assert_array_almost_equal(thresholds, sk_thresholds)


def test_check_compute_fn():
    y_pred = torch.zeros((8, 13))
    y_pred[:, 1] = 1
    y_true = torch.zeros_like(y_pred)
    output = (y_pred, y_true)

    em = PrecisionRecallCurve(check_compute_fn=True)

    em.reset()
    with pytest.warns(EpochMetricWarning, match=r"Probably, there can be a problem with `compute_fn`"):
        em.update(output)

    em = PrecisionRecallCurve(check_compute_fn=False)
    em.update(output)


def _test_distrib_binary_input(device):

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(y_pred, y, batch_size, metric_device):

        metric_device = torch.device(metric_device)
        prc = PrecisionRecallCurve(device=metric_device)

        torch.manual_seed(10 + rank)

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
        assert PrecisionRecallCurve(np_y, np_y_pred) == pytest.approx(res)

@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):

    device = idist.device()
    _test_distrib_binary_input(device)
@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):

    device = idist.device()
    _test_distrib_binary_input(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_binary_input, (device,), np=nproc, do_init=True)

@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):

    device = idist.device()
    _test_distrib_binary_input(device)

@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):

    device = idist.device()
    _test_distrib_binary_input(device)

@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_binary_input(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_binary_input(device)