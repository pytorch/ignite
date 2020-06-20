import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from ignite.contrib.metrics import ROC_AUC
from ignite.engine import Engine
from ignite.metrics.epoch_metric import EpochMetricWarning


def test_roc_auc_score():

    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2 :] = 1
    np_roc_auc = roc_auc_score(np_y, np_y_pred)

    roc_auc_metric = ROC_AUC()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    roc_auc_metric.reset()
    roc_auc_metric.update((y_pred, y))
    roc_auc = roc_auc_metric.compute()

    assert roc_auc == np_roc_auc


def test_roc_auc_score_2():

    np.random.seed(1)
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2 :] = 1
    np.random.shuffle(np_y)
    np_roc_auc = roc_auc_score(np_y, np_y_pred)

    roc_auc_metric = ROC_AUC()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    roc_auc_metric.reset()
    n_iters = 10
    batch_size = size // n_iters
    for i in range(n_iters):
        idx = i * batch_size
        roc_auc_metric.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

    roc_auc = roc_auc_metric.compute()

    assert roc_auc == np_roc_auc


def test_integration_roc_auc_score_with_output_transform():

    np.random.seed(1)
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2 :] = 1
    np.random.shuffle(np_y)

    np_roc_auc = roc_auc_score(np_y, np_y_pred)

    batch_size = 10

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    roc_auc_metric = ROC_AUC(output_transform=lambda x: (x[1], x[2]))
    roc_auc_metric.attach(engine, "roc_auc")

    data = list(range(size // batch_size))
    roc_auc = engine.run(data, max_epochs=1).metrics["roc_auc"]

    assert roc_auc == np_roc_auc


def test_integration_roc_auc_score_with_activated_output_transform():

    np.random.seed(1)
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y_pred_sigmoid = torch.sigmoid(torch.from_numpy(np_y_pred)).numpy()
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2 :] = 1
    np.random.shuffle(np_y)

    np_roc_auc = roc_auc_score(np_y, np_y_pred_sigmoid)

    batch_size = 10

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    roc_auc_metric = ROC_AUC(output_transform=lambda x: (torch.sigmoid(x[1]), x[2]))
    roc_auc_metric.attach(engine, "roc_auc")

    data = list(range(size // batch_size))
    roc_auc = engine.run(data, max_epochs=1).metrics["roc_auc"]

    assert roc_auc == np_roc_auc


def test_check_compute_fn():
    y_pred = torch.zeros((8, 13))
    y_pred[:, 1] = 1
    y_true = torch.zeros_like(y_pred)
    output = (y_pred, y_true)

    em = ROC_AUC(check_compute_fn=True)

    em.reset()
    with pytest.warns(EpochMetricWarning, match=r"Probably, there can be a problem with `compute_fn`"):
        em.update(output)

    em = ROC_AUC(check_compute_fn=False)
    em.update(output)
