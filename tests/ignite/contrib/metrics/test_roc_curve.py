import numpy as np
from sklearn.metrics import roc_curve

import torch

from ignite.contrib.metrics.roc_curve import RocCurve
from ignite.engine import Engine


def test_roc_curve():
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2:] = 1
    sk_fpr, sk_tpr, sk_thresholds = roc_curve(np_y, np_y_pred)

    roc_curve_metric = RocCurve()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    roc_curve_metric.update((y_pred, y))
    fpr, tpr, thresholds = roc_curve_metric.compute()

    assert np.array_equal(fpr, sk_fpr)
    assert np.array_equal(tpr, sk_tpr)
    # assert thresholds almost equal, due to numpy->torch->numpy conversion
    np.testing.assert_array_almost_equal(thresholds, sk_thresholds)


def test_integration_roc_curve_with_output_transform():
    np.random.seed(1)
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2:] = 1
    np.random.shuffle(np_y)

    sk_fpr, sk_tpr, sk_thresholds = roc_curve(np_y, np_y_pred)

    batch_size = 10

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx: idx + batch_size]
        y_pred_batch = np_y_pred[idx: idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    roc_curve_metric = RocCurve(output_transform=lambda x: (x[1], x[2]))
    roc_curve_metric.attach(engine, "roc_curve")

    data = list(range(size // batch_size))
    fpr, tpr, thresholds = engine.run(data, max_epochs=1).metrics["roc_curve"]

    assert np.array_equal(fpr, sk_fpr)
    assert np.array_equal(tpr, sk_tpr)
    # assert thresholds almost equal, due to numpy->torch->numpy conversion
    np.testing.assert_array_almost_equal(thresholds, sk_thresholds)


def test_integration_roc_curve_with_activated_output_transform():
    np.random.seed(1)
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y_pred_sigmoid = torch.sigmoid(torch.from_numpy(np_y_pred)).numpy()
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2:] = 1
    np.random.shuffle(np_y)

    sk_fpr, sk_tpr, sk_thresholds = roc_curve(np_y, np_y_pred_sigmoid)

    batch_size = 10

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx: idx + batch_size]
        y_pred_batch = np_y_pred[idx: idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    roc_curve_metric = RocCurve(output_transform=lambda x: (torch.sigmoid(x[1]), x[2]))
    roc_curve_metric.attach(engine, "roc_curve")

    data = list(range(size // batch_size))
    fpr, tpr, thresholds = engine.run(data, max_epochs=1).metrics["roc_curve"]

    assert np.array_equal(fpr, sk_fpr)
    assert np.array_equal(tpr, sk_tpr)
    # assert thresholds almost equal, due to numpy->torch->numpy conversion
    np.testing.assert_array_almost_equal(thresholds, sk_thresholds)
