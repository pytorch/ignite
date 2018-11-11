from __future__ import division
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
import torch


class MeanAbsoluteRelativeError(Metric):
    """
    Calculate Mean Absolute Relative Error

    - `update` must receive output of the form `(y_pred, y)`

    """

    def reset(self):
        self._sum_of_absolute_relative_errors = 0.0
        self._num_samples = 0

    def update(self, output):
        y_pred, y = output
        absolute_error = torch.abs(y_pred - y.view_as(y_pred)) / torch.abs(y.view_as(y_pred))
        self._sum_of_absolute_relative_errors += torch.sum(absolute_error).item()
        self._num_samples += y.shape[0]

    def compute(self):
        if self._num_samples == 0:
            raise NotComputableError('MeanAbsoluteError must have at least one example before it can be computed')
        return self._sum_of_absolute_relative_errors / self._num_samples
