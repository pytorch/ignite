from typing import Callable

import torch

from ignite.metrics import EpochMetric


class CohenKappa(EpochMetric):
    """Compute different types of Cohen's Kappa: Non-Wieghted, Linear, Quadratic.
    Accumulating predictions and the ground-truth during an epoch and applying
    `sklearn.metrics.cohen_kappa_score <https://scikit-learn.org/stable/modules/
    generated/sklearn.metrics.cohen_kappa_score.html>`_ .

    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        weights (str): a string is used to define the type of Cohen's Kappa whether Non-Weighted or Linear or Quadratic
            (default: "")
        check_compute_fn (bool): Default False. If True, `cohen_kappa_score
            <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html>`_
            is run on the first batch of data to ensure there are
            no issues. User will be warned in case there are any issues computing the function.

    .. code-block:: python

        def activated_output_transform(output):
            y_pred, y = output
            return y_pred, y

        weights = None or linear or quadratic

        cohen_kappa = CohenKappa(activated_output_transform, weights)

    """

    def __init__(self, output_transform: Callable = lambda x: x, weights: str = "", check_compute_fn: bool = False):
        # initalize weights
        if weights == "":
            self.weights = None
        else:
            self.weights = weights

        self.cohen_kappa_compute = self.get_cohen_kappa_fn()

        super(CohenKappa, self).__init__(
            self.cohen_kappa_compute, output_transform=output_transform, check_compute_fn=check_compute_fn
        )

    def get_cohen_kappa_fn(self) -> float:

        try:
            from sklearn.metrics import cohen_kappa_score
        except ImportError:
            raise RuntimeError("This contrib module requires sklearn to be installed.")

        def wrapper(y_targets: torch.Tensor, y_preds: torch.Tensor) -> float:
            y_true = y_targets.numpy()
            y_pred = y_preds.numpy()
            return cohen_kappa_score(y_true, y_pred, weights=self.weights)

        return wrapper
