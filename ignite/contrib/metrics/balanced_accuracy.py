from typing import Callable

import torch

from ignite.metrics import EpochMetric


def balanced_accuracy_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    try:
        from sklearn.metrics import balanced_accuracy_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()
    return balanced_accuracy_score(y_true, y_pred)


class BalancedAccuracy(EpochMetric):
    """Computes balanced accuracy, defined as the average of recall obtained on each class,
    by accumulating predictions and the ground-truth during an epoch and applying
    `sklearn.metrics.balanced_accuracy_score <http://scikit-learn.org/stable/modules/generated/
    sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score>`_ .

    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        check_compute_fn (bool): Default False. If True, `balanced_accuracy_score
            <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
            #sklearn.metrics.balanced_accuracy_score>`_ is run on the first batch of data to ensure there are
            no issues. User will be warned in case there are any issues computing the function.

    BalancedAccuracy expects y to be comprised of 0's and 1's. y_pred must either be probability estimates or
    confidence values. To apply an activation to y_pred, use output_transform as shown below:

    .. code-block:: python

        def activated_output_transform(output):
            y_pred, y = output
            y_pred = torch.sigmoid(y_pred)
            return y_pred, y

        balanced_accuracy = BalancedAccuracy(activated_output_transform)

    """

    def __init__(self, output_transform: Callable = lambda x: x, check_compute_fn: bool = False) -> None:
        super(BalancedAccuracy, self).__init__(
            balanced_accuracy_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn
        )
