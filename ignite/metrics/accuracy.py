from __future__ import division

import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError


class Accuracy(Metric):
    """
    Calculates the accuracy.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...)
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

        if y_pred.shape[0] != y.shape[0]:
            raise ValueError("y and y_pred must be of same length or batch size.")

        if y.ndimension() > 1:

            y_shape = list(y.shape)[1:]
            y_pred_shape = list(y_pred.shape)[-len(y_shape):]
            pred_dim = list(y_pred.shape)[:-len(y_shape)]

            if y_shape != y_pred_shape:
                raise ValueError("y must have shape of (batch_size, ...) " +
                                 "and y_pred must have shape of (batch_size, num_classes, ...) " +
                                 "(batch_size, ...).")

            if len(pred_dim) == 2:
                if pred_dim[1] > 1:
                    is_categorical = True
                else:
                    is_categorical = False
            elif len(pred_dim) == 1:
                is_categorical = False
            else:
                raise ValueError("y must have shape of (batch_size, ...) " +
                                 "and y_pred must have shape of (batch_size, num_classes, ...) " +
                                 "(batch_size, ...).")
        else:
            if y_pred.ndimension() == 2:
                is_categorical = True
            elif y_pred.ndimension() == 1:
                is_categorical = False
            else:
                raise ValueError("y must have shape of (batch_size, ...) " +
                                 "and y_pred must have shape of (batch_size, num_classes, ...) " +
                                 "(batch_size, ...).")
        if is_categorical:
            indices = torch.max(y_pred, 1)[1]
            correct = torch.eq(indices, y).view(-1)
        else:
            correct = torch.eq(torch.round(y_pred).type(y.type()), y).view(-1)

        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples
