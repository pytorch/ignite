import numpy as np
from sklearn.metrics import average_precision_score

import torch

from ignite.engine import Engine
from ignite.contrib.metrics import AveragePrecision


def test_ap_score():

    size = 100
    np_y_pred = np.random.rand(size, 5)
    np_y = np.random.randint(0, 2, size=(size, 5), dtype=np.long)
    np_ap = average_precision_score(np_y, np_y_pred)

    ap_metric = AveragePrecision()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    ap_metric.reset()
    ap_metric.update((y_pred, y))
    ap = ap_metric.compute()

    assert ap == np_ap


def test_ap_score_2():

    np.random.seed(1)
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2:] = 1
    np.random.shuffle(np_y)
    np_ap = average_precision_score(np_y, np_y_pred)

    ap_metric = AveragePrecision()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    ap_metric.reset()
    n_iters = 10
    batch_size = size // n_iters
    for i in range(n_iters):
        idx = i * batch_size
        ap_metric.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

    ap = ap_metric.compute()

    assert ap == np_ap


def test_integration_ap_score_with_output_transform():

    np.random.seed(1)
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2:] = 1
    np.random.shuffle(np_y)

    np_ap = average_precision_score(np_y, np_y_pred)

    batch_size = 10

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx:idx + batch_size]
        y_pred_batch = np_y_pred[idx:idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    ap_metric = AveragePrecision(output_transform=lambda x: (x[1], x[2]))
    ap_metric.attach(engine, 'ap')

    data = list(range(size // batch_size))
    ap = engine.run(data, max_epochs=1).metrics['ap']

    assert ap == np_ap


def test_integration_ap_score_with_activated_output_transform():

    np.random.seed(1)
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y_pred_softmax = torch.softmax(torch.from_numpy(np_y_pred), dim=1).numpy()
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2:] = 1
    np.random.shuffle(np_y)

    np_ap = average_precision_score(np_y, np_y_pred_softmax)

    batch_size = 10

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx:idx + batch_size]
        y_pred_batch = np_y_pred[idx:idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    ap_metric = AveragePrecision(output_transform=lambda x: (torch.softmax(x[1], dim=1), x[2]))
    ap_metric.attach(engine, 'ap')

    data = list(range(size // batch_size))
    ap = engine.run(data, max_epochs=1).metrics['ap']

    assert ap == np_ap
