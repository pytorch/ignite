from __future__ import division

import torch

from ignite.exceptions import NotComputableError
from ignite.contrib.metrics.regression._base import _BaseRegression


class MeanError(_BaseRegression):
    r"""
    Calculates the Mean Error:

    :math:`\text{ME} = \frac{1}{n}\sum_{j=1}^n (A_j - P_j)`,

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    More details can be found in the reference `Botchkarev 2018`__.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)`.

    __ https://arxiv.org/abs/1809.03006

    """
    def reset(self):
        self._sum_of_errors = 0.0
        self._num_examples = 0

    def _update(self, output):
        y_pred, y = output
        errors = (y.view_as(y_pred) - y_pred)
        self._sum_of_errors += torch.sum(errors).item()
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MeanError must have at least one example before it can be computed')
        return self._sum_of_errors / self._num_examples
