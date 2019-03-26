from __future__ import division

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


class MeanAbsoluteError(Metric):
    """
    Calculates the mean absolute error.

    - `update` must receive output of the form `(y_pred, y)`.
    """
    def reset(self):
        self._sum_of_absolute_errors = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        absolute_errors = torch.abs(y_pred - y.view_as(y_pred))
        self._sum_of_absolute_errors += torch.sum(absolute_errors).item()
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MeanAbsoluteError must have at least one example before it can be computed.')
        return self._sum_of_absolute_errors / self._num_examples
