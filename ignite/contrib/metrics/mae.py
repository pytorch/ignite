from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
import torch


class MaximumAbsoluteError(Metric):
    """
    Calculates the maximum absolute error.

    - `update` must receive output of the form `(y_pred, y)`.
    """
    def reset(self):
        self._max_of_absolute_errors = -1

    def update(self, output):
        y_pred, y = output
        mae = torch.abs(y_pred - y.view_as(y_pred)).max().item()
        if self._max_of_absolute_errors < mae:
            self._max_of_absolute_errors = mae

    def compute(self):
        if self._max_of_absolute_errors < 0:
            raise NotComputableError('MaximumAbsoluteError must have at least one example before it can be computed')
        return self._max_of_absolute_errors
