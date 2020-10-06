from typing import Tuple

import torch

from ignite.contrib.metrics.regression._base import _BaseRegression
from ignite.exceptions import NotComputableError


class MeanAbsoluteRelativeError(_BaseRegression):
    r"""
    Calculate Mean Absolute Relative Error:

    :math:`\text{MARE} = \frac{1}{n}\sum_{j=1}^n\frac{\left|A_j-P_j\right|}{\left|A_j\right|}`,

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    More details can be found in the reference `Botchkarev 2018`__.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)`.

    __ https://arxiv.org/ftp/arxiv/papers/1809/1809.03006.pdf

    """

    def reset(self):
        self._sum_of_absolute_relative_errors = 0.0
        self._num_samples = 0

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        y_pred, y = output
        if (y == 0).any():
            raise NotComputableError("The ground truth has 0.")
        absolute_error = torch.abs(y_pred - y.view_as(y_pred)) / torch.abs(y.view_as(y_pred))
        self._sum_of_absolute_relative_errors += torch.sum(absolute_error).item()
        self._num_samples += y.size()[0]

    def compute(self):
        if self._num_samples == 0:
            raise NotComputableError(
                "MeanAbsoluteRelativeError must have at least" "one sample before it can be computed."
            )
        return self._sum_of_absolute_relative_errors / self._num_samples
