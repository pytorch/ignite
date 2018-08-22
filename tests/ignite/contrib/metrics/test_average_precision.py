import numpy as np
from sklearn.metrics import average_precision_score

import torch

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


def test_ap_score_with_activation():

    size = 100
    np_y_pred = np.random.rand(size, 5)
    np_y_pred_softmax = torch.softmax(torch.from_numpy(np_y_pred), dim=1).numpy()
    np_y = np.random.randint(0, 2, size=(size, 5), dtype=np.long)
    np_ap = average_precision_score(np_y, np_y_pred_softmax)

    ap_metric = AveragePrecision(activation=torch.nn.Softmax(dim=1))
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    ap_metric.reset()
    ap_metric.update((y_pred, y))
    ap = ap_metric.compute()

    assert ap == np_ap
