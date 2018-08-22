from functools import partial

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    raise RuntimeError("This contrib module requires sklearn to be installed")

from ignite.metrics import EpochMetric


def roc_auc_compute_fn(y_preds, y_targets, activation=None):
    y_true = y_targets.numpy()
    if activation is not None:
        y_preds = activation(y_preds)
    y_pred = y_preds.numpy()
    return roc_auc_score(y_true, y_pred)


class ROC_AUC(EpochMetric):
    """Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    accumulating predictions and the ground-truth during an epoch and applying
    `sklearn.metrics.roc_auc_score <http://scikit-learn.org/stable/modules/generated/
    sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score>`_

    Args:
        activation (Callable, optional): optional function to apply on prediction tensors,
            e.g. `activation=torch.sigmoid` to transform logits.
    """
    def __init__(self, activation=None):
        super(ROC_AUC, self).__init__(partial(roc_auc_compute_fn, activation=activation))
