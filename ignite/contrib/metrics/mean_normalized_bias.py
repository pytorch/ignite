from __future__ import division

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


class MeanNormalizedBias(Metric):
    """
    Calculates the mean normalized bias.

    - `update` must receive output of the form `(y_pred, y)`.
    """
    def reset(self):
        self._sum_of_errors = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output

        if (y == 0).any():
            raise NotComputableError('The ground truth has 0.')

        errors = (y_pred - y.view_as(y_pred)) / y
        self._sum_of_errors += torch.sum(errors).item()
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MeanNormalizedBias must have at least one example before it can be computed')
        return self._sum_of_errors / self._num_examples
