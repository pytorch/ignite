
from functools import partial

from sklearn.metrics import roc_auc_score

from ignite.metrics import EpochMetric


def roc_auc_compute_fn(y_preds, y_targets, activation=None):
    y_true = y_targets.numpy()
    if activation is not None:
        y_preds = activation(y_preds)
    y_pred = y_preds.numpy()
    return roc_auc_score(y_true, y_pred)


class ROC_AUC(EpochMetric):

    def __init__(self, activation=None):
        super(ROC_AUC, self).__init__(partial(roc_auc_compute_fn, activation=activation))
