import torch

from ignite.contrib.metrics.regression._base import _BaseRegression
from ignite.exceptions import NotComputableError


class R2Score(_BaseRegression):
    r"""
        Calculates the R-Squared, the
        `coefficient of determination <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_:

        :math:`R^2 = 1 - \frac{\sum_{j=1}^n(A_j - P_j)^2}{\sum_{j=1}^n(A_j - \bar{A})^2}`,

        where :math:`A_j` is the ground truth, :math:`P_j` is the predicted value and
        :math:`\bar{A}` is the mean of the ground truth.

        - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)` and of type `float32`.
    """

    def reset(self):
        self._num_examples = 0
        self._sum_of_errors = 0
        self._y_sq_sum = 0
        self._y_sum = 0

    def _update(self, output):
        y_pred, y = output
        self._num_examples += y.shape[0]
        self._sum_of_errors += torch.sum(torch.pow(y_pred - y, 2)).item()

        self._y_sum += torch.sum(y).item()
        self._y_sq_sum += torch.sum(torch.pow(y, 2)).item()

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("R2Score must have at least one example before it can be computed.")
        return 1 - self._sum_of_errors / (self._y_sq_sum - (self._y_sum ** 2) / self._num_examples)
