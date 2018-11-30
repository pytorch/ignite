from __future__ import division

import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError


class _BaseClassification(Metric):
    def __init__(self, output_transform=lambda x: x, threshold_function=None):
        if threshold_function is not None:
            if callable(threshold_function):
                self._threshold = threshold_function
            else:
                raise ValueError("threshold_function must be a callable function.")
        else:
            self._threshold = threshold_function
        self._updated = False
        super(_BaseClassification, self).__init__(output_transform=output_transform)

    def _check_shape(self, output):
        y_pred, y = output

        if y.ndimension() > 1 and y.shape[1] == 1:
            y = y.squeeze(dim=1)

        if y_pred.ndimension() > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=1)

        if not (y.ndimension() == y_pred.ndimension() or y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError("y must have shape of (batch_size, ...) and y_pred must have "
                             "shape of (batch_size, num_categories, ...) or (batch_size, ...).")

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if not (y_shape == y_pred_shape):
            raise ValueError("y and y_pred must have compatible shapes.")

        return y_pred, y

    def _check_type(self, output):
        y_pred, y = output

        if y.ndimension() + 1 == y_pred.ndimension():
            update_type = 'multiclass'
        elif (y_pred.shape == y.shape) or y_pred.shape[1] == 1:
            update_type = 'binary'
            if not self._updated and self._threshold is None:
                self._threshold = torch.round
        else:
            raise TypeError('Invalid shapes of y (shape={}) and y_pred (shape={}), check documentation'
                            ' for expected shapes of y and y_pred.'.format(y.shape, y_pred.shape))
        if not self._updated:
            self._type = update_type
            self._updated = True
        else:
            if self._type != update_type:
                raise TypeError('update_type has changed from {} to {}.'.format(self._type, update_type))


class Accuracy(_BaseClassification):
    """
    Calculates the accuracy.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...)
    - `y` must be in the following shape (batch_size, ...)
    """

    def __init__(self, output_transform=lambda x: x, threshold_function=None):
        super(Accuracy, self).__init__(output_transform, threshold_function)

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):

        y_pred, y = self._check_shape(output)
        self._check_type((y_pred, y))

        y_pred, y = output

        if self._type == 'binary':
            y_pred = y_pred.view(-1)
            y = y.view(-1)

            y_pred = self._threshold(y_pred)
            if not torch.equal(y, y ** 2):
                raise ValueError("For binary cases, y must contain 0's and 1's only.")
            if not torch.equal(y_pred, y_pred ** 2):
                raise ValueError("For binary cases, y_pred must contain 0's and 1's only.")
            correct = torch.eq(y_pred, y.type_as(y_pred)).view(-1)
        else:
            indices = torch.max(y_pred, dim=1)[1]
            correct = torch.eq(indices, y.type_as(indices)).view(-1)

        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples
