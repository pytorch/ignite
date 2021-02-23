import numpy as np
import pytest
import torch
from sklearn.metrics import cohen_kappa_score

from ignite.contrib.metrics import CohenKappa
from ignite.engine import Engine


def test_cohen_kappa_non_weighted():

    size = 100
    np_y_pred = np.random.randint(0, 2, size=(size, 1), dtype=np.long)
    np_y = np.random.randint(0, 2, size=(size, 1), dtype=np.long)
    np_ck = cohen_kappa_score(np_y, np_y_pred)

    ck_metric = CohenKappa()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    ck_metric.reset()
    ck_metric.update((y_pred, y))
    ck = ck_metric.compute()

    assert ck == np_ck


def test_cohen_kappa_linear_weighted():

    size = 100
    np_y_pred = np.random.randint(0, 2, size=(size, 1), dtype=np.long)
    np_y = np.random.randint(0, 2, size=(size, 1), dtype=np.long)
    np_ck = cohen_kappa_score(np_y, np_y_pred, weights="linear")

    ck_metric = CohenKappa(weights="linear")
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    ck_metric.reset()
    ck_metric.update((y_pred, y))
    ck = ck_metric.compute()

    assert ck == np_ck


def test_cohen_kappa_queadratic_weighted():

    size = 100
    np_y_pred = np.random.randint(0, 2, size=(size, 1), dtype=np.long)
    np_y = np.random.randint(0, 2, size=(size, 1), dtype=np.long)
    np_ck = cohen_kappa_score(np_y, np_y_pred, weights="quadratic")

    ck_metric = CohenKappa(weights="quadratic")
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    ck_metric.reset()
    ck_metric.update((y_pred, y))
    ck = ck_metric.compute()

    assert ck == np_ck


def test_integration_cohen_kappa_non_weighted_with_output_transform():
    np.random.seed(1)
    size = 100
    np_y_pred = np.random.randint(0, 2, size=(size, 1), dtype=np.long)
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2 :] = 1
    np.random.shuffle(np_y)

    ck_value_sk = cohen_kappa_score(np_y, np_y_pred)

    batch_size = 10

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    ck_metric = CohenKappa(output_transform=lambda x: (x[1], x[2]))
    ck_metric.attach(engine, "cohen_kappa")

    data = list(range(size // batch_size))
    ck_value = engine.run(data, max_epochs=1).metrics["cohen_kappa"]

    assert np.array_equal(ck_value, ck_value_sk)


def test_integration_cohen_kappa_linear_weighted_with_output_transform():
    np.random.seed(1)
    size = 100
    np_y_pred = np.random.randint(0, 2, size=(size, 1), dtype=np.long)
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2 :] = 1
    np.random.shuffle(np_y)

    ck_value_sk = cohen_kappa_score(np_y, np_y_pred, weights="linear")

    batch_size = 10

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    ck_metric = CohenKappa(output_transform=lambda x: (x[1], x[2]), weights="linear")
    ck_metric.attach(engine, "cohen_kappa")

    data = list(range(size // batch_size))
    ck_value = engine.run(data, max_epochs=1).metrics["cohen_kappa"]

    assert np.array_equal(ck_value, ck_value_sk)


def test_integration_cohen_kappa_quadratic_weighted_with_output_transform():
    np.random.seed(1)
    size = 100
    np_y_pred = np.random.randint(0, 2, size=(size, 1), dtype=np.long)
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2 :] = 1
    np.random.shuffle(np_y)

    ck_value_sk = cohen_kappa_score(np_y, np_y_pred, weights="quadratic")

    batch_size = 10

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    ck_metric = CohenKappa(output_transform=lambda x: (x[1], x[2]), weights="quadratic")
    ck_metric.attach(engine, "cohen_kappa")

    data = list(range(size // batch_size))
    ck_value = engine.run(data, max_epochs=1).metrics["cohen_kappa"]

    assert np.array_equal(ck_value, ck_value_sk)
