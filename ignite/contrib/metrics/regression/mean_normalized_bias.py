import torch

from ignite.contrib.metrics.regression._base import _BaseRegression
from ignite.exceptions import NotComputableError


class MeanNormalizedBias(_BaseRegression):
    r"""
    Calculates the Mean Normalized Bias:

    :math:`\text{MNB} = \frac{1}{n}\sum_{j=1}^n\frac{A_j - P_j}{A_j}`,

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    More details can be found in the reference `Botchkarev 2018`__.

    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)`.

    __ https://arxiv.org/abs/1809.03006

    """

    def reset(self):
        self._sum_of_errors = 0.0
        self._num_examples = 0

    def _update(self, output):
        y_pred, y = output

        if (y == 0).any():
            raise NotComputableError("The ground truth has 0.")

        errors = (y.view_as(y_pred) - y_pred) / y
        self._sum_of_errors += torch.sum(errors).item()
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("MeanNormalizedBias must have at least one example before it can be computed.")
        return self._sum_of_errors / self._num_examples
