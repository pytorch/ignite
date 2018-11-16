from __future__ import division

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


class FractionalBias(Metric):
    r"""
    Calculates the Fractional Bias.

    It has been proposed in `Botchkarev 2018`__.

    :math:`\text{FB} = \frac{1}{n}\sum _j^n\frac{2 * (A_j - P_j)}{A_j + P_j}`

    Where, :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y` and `y_pred` must be of same shape.

    __ https://arxiv.org/abs/1809.03006

    """
    def reset(self):
        self._sum_of_errors = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        errors = 2 * (y.view_as(y_pred) - y_pred) / (y_pred + y.view_as(y_pred))
        self._sum_of_errors += torch.sum(errors).item()
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('FractionalBias must have at least one example before it can be computed')
        return self._sum_of_errors / self._num_examples
