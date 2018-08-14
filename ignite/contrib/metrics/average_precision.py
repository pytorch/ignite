from functools import partial

from sklearn.metrics import average_precision_score

from ignite.metrics import EpochMetric


def average_precision_compute_fn(y_preds, y_targets, activation=None):
    y_true = y_targets.numpy()
    if activation is not None:
        y_preds = activation(y_preds)
    y_pred = y_preds.numpy()
    return average_precision_score(y_true, y_pred)


class AveragePrecision(EpochMetric):

    def __init__(self, activation=None):
        super(AveragePrecision, self).__init__(partial(average_precision_compute_fn, activation=activation))
