import torch

from ignite.exceptions import NotComputableError
from ignite.contrib.metrics.regression._base import _BaseRegression


class MaximumAbsoluteError(_BaseRegression):
    r"""
    Calculates the Maximum Absolute Error:

    :math:`\text{MaxAE} = \max_{j=1,n} \left( \lvert A_j-P_j \rvert \right)`,

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    More details can be found in `Botchkarev 2018`__.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)`.

    __ https://arxiv.org/abs/1809.03006

    """

    def reset(self):
        self._max_of_absolute_errors = -1

    def _update(self, output):
        y_pred, y = output
        mae = torch.abs(y_pred - y.view_as(y_pred)).max().item()
        if self._max_of_absolute_errors < mae:
            self._max_of_absolute_errors = mae

    def compute(self):
        if self._max_of_absolute_errors < 0:
            raise NotComputableError('MaximumAbsoluteError must have at least one example before it can be computed')
        return self._max_of_absolute_errors
