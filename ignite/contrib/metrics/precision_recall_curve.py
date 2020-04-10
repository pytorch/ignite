from ignite.metrics import EpochMetric


def precision_recall_curve_compute_fn(y_preds, y_targets):
    try:
        from sklearn.metrics import precision_recall_curve
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()
    return precision_recall_curve(y_true, y_pred)


class PrecisionRecallCurve(EpochMetric):
    """Compute precision-recall pairs for different probability thresholds for binary classification task
    by accumulating predictions and the ground-truth during an epoch and applying
    `sklearn.metrics.precision_recall_curve <http://scikit-learn.org/stable/modules/generated/
    sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve>`_ .

    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.

    PrecisionRecallCurve expects y to be comprised of 0's and 1's. y_pred must either be probability estimates
    or confidence values. To apply an activation to y_pred, use output_transform as shown below:

    .. code-block:: python

        def activated_output_transform(output):
            y_pred, y = output
            y_pred = torch.sigmoid(y_pred)
            return y_pred, y

        roc_auc = PrecisionRecallCurve(activated_output_transform)

    """

    def __init__(self, output_transform=lambda x: x):
        super(PrecisionRecallCurve, self).__init__(precision_recall_curve_compute_fn, output_transform=output_transform)
