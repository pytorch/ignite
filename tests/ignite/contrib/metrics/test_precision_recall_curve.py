from unittest.mock import patch

import numpy as np
import pytest
import sklearn
import torch
from sklearn.metrics import precision_recall_curve

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
