import numpy as np
from sklearn.metrics import average_precision_score

import torch

from ignite.contrib.metrics import AveragePrecision


def test_roc_auc_score():

    size = 100
    np_y_pred = np.random.randint(0, 10, size=(size,)).astype(np.float32)
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2:] = 1
    np_ap = average_precision_score(np_y, np_y_pred)

    ap_metric = AveragePrecision()
    y_pred = torch.from_numpy(np_y_pred).unsqueeze(dim=1)
    y = torch.from_numpy(np_y)

    ap_metric.reset()
    ap_metric.update((y_pred, y))
    ap = ap_metric.compute()

    assert ap == np_ap
