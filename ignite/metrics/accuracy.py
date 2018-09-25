from __future__ import division

import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError


class Accuracy(Metric):
    """
    Calculates the accuracy.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...)
    - `y` must be in the following shape (batch_size, ...)
    """
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output

        if y.ndimension() == 2 and y.shape[1] == 1:
            y = y.squeeze(dim=-1)

        if y_pred.ndimension() == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=-1)

        y_dim = list(y.shape)[1:]
        pred_dim = list(y_pred.shape)[:-len(y_dim)]

        if len(pred_dim) == 1 or y_pred.ndimension() == 1:
            correct = torch.eq(torch.round(y_pred).type(y.type()), y).view(-1)
        else:
            indices = torch.max(y_pred, 1)[1]
            correct = torch.eq(indices, y).view(-1)

        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples
