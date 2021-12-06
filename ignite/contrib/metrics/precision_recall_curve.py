from typing import Any, Callable, Tuple

import torch

from ignite.metrics import EpochMetric


def precision_recall_curve_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> Tuple[Any, Any, Any]:
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
    `sklearn.metrics.precision_recall_curve <https://scikit-learn.org/stable/modules/generated/
    sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve>`_ .

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        check_compute_fn: Default False. If True, `precision_recall_curve
            <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
            #sklearn.metrics.precision_recall_curve>`_ is run on the first batch of data to ensure there are
            no issues. User will be warned in case there are any issues computing the function.

    Note:
        PrecisionRecallCurve expects y to be comprised of 0's and 1's. y_pred must either be probability estimates
        or confidence values. To apply an activation to y_pred, use output_transform as shown below:

        .. code-block:: python

            def sigmoid_output_transform(output):
                y_pred, y = output
                y_pred = torch.sigmoid(y_pred)
                return y_pred, y
            avg_precision = PrecisionRecallCurve(sigmoid_output_transform)

    Examples:
        .. testcode::

            y_pred = torch.tensor([0.0474, 0.5987, 0.7109, 0.9997])
            y_true = torch.tensor([0, 0, 1, 1])
            prec_recall_curve = PrecisionRecallCurve()
            prec_recall_curve.attach(default_evaluator, 'prec_recall_curve')
            state = default_evaluator.run([[y_pred, y_true]])

            print("Precision", [round(i, 4) for i in state.metrics['prec_recall_curve'][0].tolist()])
            print("Recall", [round(i, 4) for i in state.metrics['prec_recall_curve'][1].tolist()])
            print("Thresholds", [round(i, 4) for i in state.metrics['prec_recall_curve'][2].tolist()])

        .. testoutput::

            Precision [1.0, 1.0, 1.0]
            Recall [1.0, 0.5, 0.0]
            Thresholds [0.7109, 0.9997]

    """

    def __init__(self, output_transform: Callable = lambda x: x, check_compute_fn: bool = False) -> None:
        super(PrecisionRecallCurve, self).__init__(
            precision_recall_curve_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn
        )
