from __future__ import division

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class MeanSquaredError(Metric):
    """
    Calculates the mean squared error.

    - `update` must receive output of the form `(y_pred, y)`.
    """
    @reinit__is_reduced
    def reset(self):
        self._sum_of_squared_errors = 0.0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        squared_errors = torch.pow(y_pred - y.view_as(y_pred), 2)
        self._sum_of_squared_errors += torch.sum(squared_errors).item()
        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_of_squared_errors", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MeanSquaredError must have at least one example before it can be computed.')
        return self._sum_of_squared_errors / self._num_examples
