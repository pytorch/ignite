import numpy as np
from sklearn.metrics import roc_auc_score

import torch

from ignite.contrib.metrics import ROC_AUC


def test_roc_auc_score():

    size = 100
    np_y_pred = np.random.randint(0, 10, size=(size,)).astype(np.float32)
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2:] = 1
    np_roc_auc = roc_auc_score(np_y, np_y_pred)

    roc_auc_metric = ROC_AUC()
    y_pred = torch.from_numpy(np_y_pred).unsqueeze(dim=1)
    y = torch.from_numpy(np_y)

    roc_auc_metric.reset()
    roc_auc_metric.update((y_pred, y))
    roc_auc = roc_auc_metric.compute()

    assert roc_auc == np_roc_auc
