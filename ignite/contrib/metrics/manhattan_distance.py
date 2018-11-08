from __future__ import division

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


class ManhattanDistance(Metric):
    """
    Calculates the Manhattan Distance.

    - `update` must receive output of the form `(y_pred, y)`.
    """
    def reset(self):
        self._sum_of_errors = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        errors = y_pred - y.view_as(y_pred)
        self._sum_of_errors += torch.sum(errors).item()

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('ManhattanDistance must have at least one example before it can be computed')
        return self._sum_of_errors
