from __future__ import division

import torch

from .metric import Metric
from ignite.exceptions import NotComputableError


class BinaryAccuracy(Metric):
    """
    Calculates the binary accuracy.

    `update` must receive output of the form (y_pred, y).
    `y_pred` must be in the following shape (batch_size, ...)
    `y` must be in the following shape (batch_size, ...)
    """
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        correct = torch.eq(torch.round(y_pred).type(torch.LongTensor), y).view(-1)
        self._num_correct += torch.sum(correct)
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('BinaryAccuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples
