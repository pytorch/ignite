from functools import partial
from ignite.metrics import EpochMetric


def roc_auc_compute_fn(y_preds, y_targets, activation=None):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed")

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
        output_transform (callable, optional): a callable that is used to transform the
            :class:`ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.

    """
    def __init__(self, activation=None, output_transform=lambda x: x):
        super(ROC_AUC, self).__init__(partial(roc_auc_compute_fn, activation=activation),
                                      output_transform=output_transform)
