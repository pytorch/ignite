from __future__ import division

import torch

from .metric import Metric
from ignite.exceptions import NotComputableError


class CategoricalAccuracy(Metric):
    """
    Calculates the categorical accuracy.

    `update` must receive output of the form (y_pred, y).
    """
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        indices = torch.max(y_pred, 1)[1]
        correct = torch.eq(indices, y)
        self._num_correct += torch.sum(correct)
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CategoricalAccuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples
