import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from ignite.contrib.metrics.precision_recall_curve import PrecisionRecallCurve
from ignite.engine import Engine


def test_precision_recall_curve():
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y = np.zeros((size,), dtype=np.long)
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
    np_y = np.zeros((size,), dtype=np.long)
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
    np_y = np.zeros((size,), dtype=np.long)
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
